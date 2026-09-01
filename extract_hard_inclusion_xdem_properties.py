#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract effective mechanical/XDEM parameters from
all_parameters_Gr_BN_100x200_edge_crack.dat.

The input is the whitespace table written by the graphene-hBN LAMMPS
0 K top-pull script. The header may start with '#'.

Outputs
-------
hard_inclusion_outputs/
  effective_hard_inclusion_properties.csv
  processed_hard_inclusion_curve.csv
  xdem_effective_hard_inclusion_params.json
  hard_inclusion_summary.txt
  stress_strain.png
  damage_evolution.png
  energy_evolution.png
  poisson_ratio.png

Important
---------
The values extracted from one graphene+hBN calculation are effective
properties of the complete cracked composite specimen. They do not uniquely
separate intrinsic graphene, hBN, and graphene/hBN interface properties.
Separate pristine graphene and pristine hBN calculations are required for a
phase-resolved heterogeneous XDEM model.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "i", "strain_y_command", "strain_y_measured", "strain_x_measured",
    "poisson_inst", "top_displacement_A", "gauge_A", "width_A", "height_A",
    "reaction_syy_GPa_corrected", "reaction_syy_GPa_raw",
    "reaction_syy_2D_Npm_corrected", "reaction_syy_2D_Npm_raw",
    "bottom_syy_GPa_corrected", "bottom_syy_2D_Npm_corrected",
    "virial_sxx_GPa_nom", "virial_syy_GPa_nom", "virial_szz_GPa_nom",
    "virial_sxy_GPa_nom", "virial_sxz_GPa_nom", "virial_syz_GPa_nom",
    "Ftop_x_eVA", "Ftop_y_eVA", "Ftop_z_eVA",
    "Fbottom_x_eVA", "Fbottom_y_eVA", "Fbottom_z_eVA",
    "Ftop_y0_eVA", "Fbottom_y0_eVA", "force_balance_y_eVA",
    "PE_total_eV", "PE_per_atom_eV", "dPE_total_eV", "dPE_per_atom_eV",
    "PE_graphene_eV", "PE_hBN_eV", "dPE_graphene_eV", "dPE_hBN_eV",
    "dPE_area_Jm2", "undercoord_total", "undercoord_initial",
    "newly_undercoord", "natoms", "nC", "nB", "nN", "nBN",
    "width0_A", "height0_A", "gauge0_A", "h_A",
    "Lx_box_A", "Ly_box_A", "Lz_box_A",
]


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))


def finite(value, default=float("nan")) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def json_safe(value):
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (str, bool)):
        return value
    return value


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = None
    for line in lines:
        text = line.strip()
        if not text.startswith("#"):
            continue
        candidate = text.lstrip("#").strip().split()
        if "strain_y_measured" in candidate and "reaction_syy_GPa_corrected" in candidate:
            header = candidate
            break

    if header is not None:
        numeric = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
        if not numeric:
            raise ValueError("No numeric rows were found.")
        df = pd.read_csv(io.StringIO("\n".join(numeric)), sep=r"\s+", names=header, engine="python")
    else:
        df = pd.read_csv(path, sep=r"\s+", comment="#", engine="python")
        if "strain_y_measured" not in df.columns:
            if df.shape[1] == len(EXPECTED_COLUMNS):
                df.columns = EXPECTED_COLUMNS
            else:
                raise ValueError(
                    f"Header was not recognized. Found {df.shape[1]} columns; "
                    f"expected {len(EXPECTED_COLUMNS)}. Keep the '# i ...' header line."
                )

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    required = [
        "strain_y_measured", "reaction_syy_GPa_corrected",
        "reaction_syy_2D_Npm_corrected",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["strain_y_measured", "reaction_syy_GPa_corrected"])
    df = df.sort_values("strain_y_measured").drop_duplicates("strain_y_measured", keep="last")
    df = df.reset_index(drop=True)
    if len(df) < 3:
        raise ValueError("At least three valid rows are required.")
    return df


def orient(values: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(values, float) - float(values[0])
    sign = -1 if abs(np.nanmin(values)) > np.nanmax(values) else 1
    return sign * values, sign


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(values, float).copy()
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy(float)


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    good = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[good], np.asarray(y)[good]
    if len(x) < 2:
        raise ValueError("Too few points for a linear fit.")
    A = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = slope * x + intercept
    residual = y - pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    if len(x) > 2:
        variance = ss_res / (len(x) - 2)
        sxx = float(np.sum((x - x.mean())**2))
        slope_se = math.sqrt(variance / sxx) if sxx > 0 else float("nan")
    else:
        slope_se = float("nan")
    return float(slope), float(intercept), float(r2), float(slope_se)


def elastic_mask(strain: np.ndarray, lo: float, hi: float, minimum: int) -> np.ndarray:
    mask = np.isfinite(strain) & (strain >= lo) & (strain <= hi)
    if mask.sum() >= minimum:
        return mask
    ids = np.flatnonzero(np.isfinite(strain) & (strain >= 0))[:max(minimum, 4)]
    mask = np.zeros(len(strain), dtype=bool)
    mask[ids] = True
    if mask.sum() < 2:
        raise ValueError("Too few points in the elastic range.")
    return mask


def first_consecutive(mask: np.ndarray, count: int) -> Optional[int]:
    run = 0
    for i, state in enumerate(mask):
        run = run + 1 if state else 0
        if run >= max(1, count):
            return i - max(1, count) + 1
    return None


def value_at(values: np.ndarray, index: Optional[int], default=float("nan")) -> float:
    if index is None or index < 0 or index >= len(values):
        return default
    return finite(values[index], default)


def extract(df: pd.DataFrame, args, source: Path) -> tuple[dict, pd.DataFrame]:
    out = df.copy()

    eps = out["strain_y_measured"].to_numpy(float)
    eps -= eps[0]
    eps_cmd = out["strain_y_command"].to_numpy(float) if "strain_y_command" in out else eps.copy()
    eps_cmd -= eps_cmd[0]
    eps_x = out["strain_x_measured"].to_numpy(float) if "strain_x_measured" in out else np.full(len(out), np.nan)
    if np.isfinite(eps_x[0]):
        eps_x -= eps_x[0]

    s3, sign = orient(out["reaction_syy_GPa_corrected"].to_numpy(float))
    s2 = out["reaction_syy_2D_Npm_corrected"].to_numpy(float)
    s2 = sign * (s2 - s2[0])
    s3s, s2s = smooth(s3, args.smooth_window), smooth(s2, args.smooth_window)

    emask = elastic_mask(eps, args.elastic_min, args.elastic_max, args.minimum_elastic_points)
    E, E0, E_r2, E_se = fit_line(eps[emask], s3[emask])
    E2D, E2D0, E2D_r2, E2D_se = fit_line(eps[emask], s2[emask])

    if np.isfinite(eps_x[emask]).sum() >= 2:
        sx, x0, x_r2, x_se = fit_line(eps[emask], eps_x[emask])
        nu = -sx
        nu_source = "linear_fit_strain_x_measured"
    else:
        nu, x0, x_r2, x_se = args.default_nu, np.nan, np.nan, np.nan
        nu_source = "default_no_lateral_strain"
    if not np.isfinite(nu) or not (-0.2 <= nu <= 0.7):
        nu, nu_source = args.default_nu, "default_unphysical_fit"

    nu_inst = np.nan
    if "poisson_inst" in out:
        vals = out.loc[emask, "poisson_inst"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            nu_inst = float(np.median(vals))

    peak = int(np.nanargmax(s3s))
    peak_stress, peak_stress_2d, peak_strain = float(s3[peak]), float(s2[peak]), float(eps[peak])

    if "newly_undercoord" in out:
        damage = out["newly_undercoord"].to_numpy(float)
        damage_mask = np.isfinite(damage) & (damage >= args.undercoord_threshold) & (eps > 0)
        damage_idx = first_consecutive(damage_mask, args.damage_consecutive)
    else:
        damage = np.full(len(out), np.nan)
        damage_idx = None

    drop_mask = np.zeros(len(out), bool)
    drop_mask[peak + 1:] = s3s[peak + 1:] <= args.stress_drop_fraction * s3s[peak]
    drop_idx = first_consecutive(drop_mask, args.drop_consecutive)

    energy_idx = None
    if "dPE_total_eV" in out:
        dpe = out["dPE_total_eV"].to_numpy(float)
        running_max = np.maximum.accumulate(np.where(np.isfinite(dpe), dpe, -np.inf))
        energy_mask = np.isfinite(dpe) & (running_max - dpe >= args.energy_drop_eV)
        energy_mask[:peak + 1] = False
        energy_idx = first_consecutive(energy_mask, args.drop_consecutive)

    if damage_idx is not None:
        fracture, fracture_source = damage_idx, "newly_undercoord"
    elif drop_idx is not None:
        fracture, fracture_source = drop_idx, "post_peak_stress_drop"
    elif energy_idx is not None:
        fracture, fracture_source = energy_idx, "potential_energy_drop"
    else:
        fracture, fracture_source = peak, "peak_stress_fallback"

    def first_col(primary: str, fallback: str, default=np.nan):
        if primary in out:
            val = finite(out[primary].iloc[0])
            if np.isfinite(val):
                return val
        if fallback in out:
            val = finite(out[fallback].iloc[0])
            if np.isfinite(val):
                return val
        return float(default)

    width = first_col("width0_A", "width_A")
    height = first_col("height0_A", "height_A")
    gauge = first_col("gauge0_A", "gauge_A")
    thickness = first_col("h_A", "h_A", args.thickness_A)
    if not np.isfinite(thickness) or thickness <= 0:
        thickness = args.thickness_A

    def integrate_to(index: int):
        return (
            trapz(np.maximum(s3[:index + 1], 0), eps[:index + 1]),
            trapz(np.maximum(s2[:index + 1], 0), eps[:index + 1]),
        )

    t_peak, w2_peak = integrate_to(peak)
    t_frac, w2_frac = integrate_to(fracture)
    t_full, w2_full = integrate_to(len(out) - 1)

    # 1 GPa * 1 Angstrom = 0.1 J/m^2.
    Gc_work_peak = t_peak * thickness * 0.1
    Gc_work_frac = t_frac * thickness * 0.1
    Gc_work_full = t_full * thickness * 0.1

    alpha = args.crack_length_A / width if np.isfinite(width) and width > 0 else np.nan
    if np.isfinite(alpha) and 0 < alpha < 1 and E > 0:
        Y = 1.12 - 0.231*alpha + 10.55*alpha**2 - 21.72*alpha**3 + 30.39*alpha**4
        K_pa = Y * peak_stress * 1e9 * math.sqrt(math.pi * args.crack_length_A * 1e-10)
        K_mpa = K_pa / 1e6
        Eprime = E*1e9/(1-nu**2) if args.plane_condition == "plane_strain" else E*1e9
        Gc_lefm = K_pa**2 / Eprime
    else:
        Y = K_mpa = Gc_lefm = np.nan

    selected = {
        "work_to_peak": Gc_work_peak,
        "work_to_fracture": Gc_work_frac,
        "lefm": Gc_lefm,
        "2d_work_to_peak": w2_peak,
    }[args.gc_method]

    l_A = args.l_A if args.l_A is not None else args.l0_norm * width

    natoms = first_col("natoms", "natoms")
    nC, nB, nN = first_col("nC", "nC"), first_col("nB", "nB"), first_col("nN", "nN")
    nBN = first_col("nBN", "nBN", nB+nN if np.isfinite(nB+nN) else np.nan)
    atom_frac = nBN/natoms if np.isfinite(nBN) and np.isfinite(natoms) and natoms > 0 else np.nan
    area_frac = math.pi*args.inclusion_radius_A**2/(width*height) if width > 0 and height > 0 else np.nan

    def point(col: str, idx: int):
        return finite(out[col].iloc[idx]) if col in out else np.nan

    stiffness_ratio = E/args.reference_graphene_E_GPa if args.reference_graphene_E_GPa else np.nan

    props = {
        "source_file": str(source), "size": args.size, "material_key": args.material_key,
        "description": "effective graphene matrix plus circular hBN hard-inclusion edge-cracked specimen",
        "geometry": "rectangular_graphene_hBN_circular_inclusion_edge_crack",
        "loading_direction": "y", "fracture_mode": "mode_I_top_pull",
        "plane_condition": args.plane_condition, "n_points": len(out), "stress_sign_multiplier": sign,
        "elastic_min": args.elastic_min, "elastic_max": args.elastic_max, "elastic_points": int(emask.sum()),
        "E_effective_GPa": E, "E_effective_MPa": E*1000, "E_effective_2D_N_m": E2D,
        "E_intercept_GPa": E0, "E_fit_R2": E_r2, "E_slope_standard_error_GPa": E_se,
        "E2D_fit_R2": E2D_r2, "E2D_slope_standard_error_N_m": E2D_se,
        "nu_effective_yx": nu, "nu_source": nu_source,
        "nu_instantaneous_elastic_median": nu_inst, "lateral_fit_intercept": x0,
        "lateral_fit_R2": x_r2, "lateral_slope_standard_error": x_se,
        "sigma_c_effective_GPa": peak_stress, "sigma_c_smoothed_GPa": float(s3s[peak]),
        "sigma_c_effective_2D_N_m": peak_stress_2d, "epsilon_c": peak_strain, "peak_index": peak,
        "fracture_onset_index": fracture, "fracture_onset_source": fracture_source,
        "fracture_onset_strain": value_at(eps, fracture), "fracture_onset_stress_GPa": value_at(s3, fracture),
        "undercoord_onset_index": damage_idx, "undercoord_onset_strain": value_at(eps, damage_idx),
        "stress_drop_index": drop_idx, "stress_drop_strain": value_at(eps, drop_idx),
        "energy_drop_index": energy_idx, "energy_drop_strain": value_at(eps, energy_idx),
        "toughness_to_peak_GPa": t_peak, "toughness_to_peak_MJ_m3": t_peak*1000,
        "toughness_to_fracture_GPa": t_frac, "toughness_full_curve_GPa": t_full,
        "work_2D_to_peak_J_m2": w2_peak, "work_2D_to_fracture_J_m2": w2_frac,
        "work_2D_full_curve_J_m2": w2_full,
        "Gc_work_to_peak_J_m2": Gc_work_peak, "Gc_work_to_fracture_J_m2": Gc_work_frac,
        "Gc_full_curve_J_m2": Gc_work_full, "Gc_LEFM_J_m2": Gc_lefm,
        "Gc_selected_J_m2": selected, "Gc_selected_method": args.gc_method,
        "Kc_edge_MPa_sqrt_m": K_mpa, "edge_geometry_factor_Y": Y,
        "dPE_area_at_peak_J_m2": point("dPE_area_Jm2", peak),
        "dPE_area_at_fracture_onset_J_m2": point("dPE_area_Jm2", fracture),
        "dPE_graphene_at_peak_eV": point("dPE_graphene_eV", peak),
        "dPE_hBN_at_peak_eV": point("dPE_hBN_eV", peak),
        "width0_A": width, "height0_A": height, "gauge0_A": gauge, "thickness_A": thickness,
        "crack_length_A": args.crack_length_A, "crack_length_fraction_width": alpha,
        "crack_y_fraction_height": args.crack_y_fraction,
        "inclusion_radius_A": args.inclusion_radius_A,
        "inclusion_center_x_fraction": args.inclusion_center_x_fraction,
        "inclusion_center_y_fraction": args.inclusion_center_y_fraction,
        "inclusion_area_fraction_geometry": area_frac,
        "natoms": natoms, "nC": nC, "nB": nB, "nN": nN, "nBN": nBN,
        "inclusion_atom_fraction": atom_frac, "l0_norm": args.l0_norm, "l_A": l_A,
        "reference_graphene_E_GPa": args.reference_graphene_E_GPa,
        "effective_stiffness_ratio_to_reference_graphene": stiffness_ratio,
        "newly_undercoord_at_peak": point("newly_undercoord", peak),
        "newly_undercoord_at_fracture": point("newly_undercoord", fracture),
        "force_balance_y_eVA_max_abs": float(np.nanmax(np.abs(out["force_balance_y_eVA"])))
        if "force_balance_y_eVA" in out else np.nan,
    }

    out["strain_y_used"] = eps
    out["strain_y_command_zeroed"] = eps_cmd
    out["strain_x_zeroed"] = eps_x
    out["stress_3D_GPa_used"] = s3
    out["stress_3D_GPa_smoothed"] = s3s
    out["stress_2D_Npm_used"] = s2
    out["stress_2D_Npm_smoothed"] = s2s
    out["used_for_elastic_fit"] = emask.astype(int)
    out["elastic_fit_stress_GPa"] = E*eps + E0
    out["elastic_fit_stress_2D_Npm"] = E2D*eps + E2D0
    out["is_peak_point"] = 0
    out.loc[peak, "is_peak_point"] = 1
    out["is_fracture_onset"] = 0
    out.loc[fracture, "is_fracture_onset"] = 1
    return props, out


def xdem_entry(p: dict) -> dict:
    return {
        "selected_size": p["size"], "material": p["material_key"], "material_key": p["material_key"],
        "description": p["description"], "geometry": p["geometry"],
        "fracture_mode": p["fracture_mode"], "loading_direction": "y",
        "plane_condition": p["plane_condition"],
        "E_GPa": p["E_effective_GPa"], "E_MPa": p["E_effective_MPa"],
        "nu": p["nu_effective_yx"], "Gc_J_m2": p["Gc_selected_J_m2"],
        "Gc_initial_J_m2": p["Gc_selected_J_m2"], "l_A": p["l_A"], "l0_norm": p["l0_norm"],
        "sigma_c_GPa": p["sigma_c_effective_GPa"],
        "sigma_c_2D_N_m": p["sigma_c_effective_2D_N_m"], "epsilon_c": p["epsilon_c"],
        "width_A": p["width0_A"], "height_A": p["height0_A"],
        "gauge_A": p["gauge0_A"], "thickness_A": p["thickness_A"],
        "crack_type": "edge_horizontal", "crack_length_A": p["crack_length_A"],
        "crack_fraction": p["crack_length_fraction_width"],
        "crack_y_fraction": p["crack_y_fraction_height"], "crack_x_start_A": 0.0,
        "inclusion_type": "circular_hBN_hard_inclusion",
        "inclusion_radius_A": p["inclusion_radius_A"],
        "inclusion_center_x_fraction": p["inclusion_center_x_fraction"],
        "inclusion_center_y_fraction": p["inclusion_center_y_fraction"],
        "inclusion_area_fraction_geometry": p["inclusion_area_fraction_geometry"],
        "Kc_edge_MPa_sqrt_m": p["Kc_edge_MPa_sqrt_m"],
        "Gc_work_to_peak_J_m2": p["Gc_work_to_peak_J_m2"],
        "Gc_LEFM_J_m2": p["Gc_LEFM_J_m2"], "Gc_selected_method": p["Gc_selected_method"],
        "note": (
            "Effective properties of the complete graphene+hBN hard-inclusion specimen. "
            "Use separate pristine graphene and hBN calculations for phase-resolved XDEM parameters."
        ),
    }


def write_summary(path: Path, p: dict):
    text = f"""Graphene-hBN hard-inclusion effective properties
================================================
Source: {p['source_file']}

E effective (3D) : {p['E_effective_GPa']:.8g} GPa
E effective (2D) : {p['E_effective_2D_N_m']:.8g} N/m
Elastic fit R2    : {p['E_fit_R2']:.8g}
Poisson ratio     : {p['nu_effective_yx']:.8g}
Peak stress       : {p['sigma_c_effective_GPa']:.8g} GPa
Peak 2D stress    : {p['sigma_c_effective_2D_N_m']:.8g} N/m
Peak strain       : {p['epsilon_c']:.8g}
Fracture onset    : {p['fracture_onset_strain']:.8g}
Onset source      : {p['fracture_onset_source']}
Gc work-to-peak   : {p['Gc_work_to_peak_J_m2']:.8g} J/m2
Kc edge estimate  : {p['Kc_edge_MPa_sqrt_m']:.8g} MPa sqrt(m)
Gc LEFM estimate  : {p['Gc_LEFM_J_m2']:.8g} J/m2
Selected XDEM Gc  : {p['Gc_selected_J_m2']:.8g} J/m2 ({p['Gc_selected_method']})

WARNING: these are effective composite-specimen properties, not separately
identified graphene, hBN, or interface properties.
"""
    path.write_text(text, encoding="utf-8")


def plots(df: pd.DataFrame, p: dict, folder: Path, args):
    eps = df["strain_y_used"].to_numpy(float)
    stress = df["stress_3D_GPa_used"].to_numpy(float)
    sm = df["stress_3D_GPa_smoothed"].to_numpy(float)
    fit = df["used_for_elastic_fit"].to_numpy(int).astype(bool)
    peak, frac = int(p["peak_index"]), int(p["fracture_onset_index"])

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.plot(eps, stress, marker="o", markersize=3, linewidth=1.2, label="LAMMPS")
    ax.plot(eps, sm, linewidth=1.3, label=f"rolling mean ({args.smooth_window})")
    ax.plot(eps[fit], df.loc[fit, "elastic_fit_stress_GPa"], linestyle="--", linewidth=1.4,
            label=f"elastic fit: E={p['E_effective_GPa']:.3f} GPa")
    ax.scatter([eps[peak]], [stress[peak]], s=55, label="peak")
    ax.scatter([eps[frac]], [stress[frac]], s=55, marker="s", label="fracture onset")
    ax.set_xlabel(r"Measured strain $\varepsilon_{yy}$")
    ax.set_ylabel(r"Corrected reaction stress $\sigma_{yy}$ (GPa)")
    ax.legend(frameon=False); ax.grid(True, alpha=0.25); fig.tight_layout()
    fig.savefig(folder/"stress_strain.png", dpi=300); plt.close(fig)

    if "newly_undercoord" in df:
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        ax.plot(eps, df["newly_undercoord"], marker="o", markersize=3, linewidth=1.2)
        ax.axvline(p["fracture_onset_strain"], linestyle="--", label="selected onset")
        ax.set_xlabel(r"Measured strain $\varepsilon_{yy}$"); ax.set_ylabel("Newly under-coordinated atoms")
        ax.legend(frameon=False); ax.grid(True, alpha=0.25); fig.tight_layout()
        fig.savefig(folder/"damage_evolution.png", dpi=300); plt.close(fig)

    energy_cols = [("dPE_total_eV", "Total"), ("dPE_graphene_eV", "Graphene"), ("dPE_hBN_eV", "h-BN")]
    present = [(c,l) for c,l in energy_cols if c in df]
    if present:
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        for col, label in present:
            ax.plot(eps, df[col], linewidth=1.3, label=label)
        ax.set_xlabel(r"Measured strain $\varepsilon_{yy}$"); ax.set_ylabel("Potential-energy change (eV)")
        ax.legend(frameon=False); ax.grid(True, alpha=0.25); fig.tight_layout()
        fig.savefig(folder/"energy_evolution.png", dpi=300); plt.close(fig)

    if "poisson_inst" in df:
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        ax.plot(eps, df["poisson_inst"], marker="o", markersize=3, linewidth=1.2)
        ax.axhline(p["nu_effective_yx"], linestyle="--", label=f"fit: {p['nu_effective_yx']:.4f}")
        ax.set_xlabel(r"Measured strain $\varepsilon_{yy}$"); ax.set_ylabel(r"Poisson ratio $\nu_{yx}$")
        ax.legend(frameon=False); ax.grid(True, alpha=0.25); fig.tight_layout()
        fig.savefig(folder/"poisson_ratio.png", dpi=300); plt.close(fig)


def parser():
    p = argparse.ArgumentParser(description="Extract hard-inclusion effective properties for XDEM.")
    p.add_argument("--input", default="all_parameters_Gr_BN_100x200_edge_crack.dat")
    p.add_argument("--output_dir", default="hard_inclusion_outputs")
    p.add_argument("--size", default="100_200")
    p.add_argument("--material_key", default="Gr_hBN_hard_inclusion_edge_crack")
    p.add_argument("--elastic_min", type=float, default=0.002)
    p.add_argument("--elastic_max", type=float, default=0.020)
    p.add_argument("--minimum_elastic_points", type=int, default=5)
    p.add_argument("--default_nu", type=float, default=0.20)
    p.add_argument("--smooth_window", type=int, default=3)
    p.add_argument("--stress_drop_fraction", type=float, default=0.80)
    p.add_argument("--drop_consecutive", type=int, default=2)
    p.add_argument("--undercoord_threshold", type=float, default=1.0)
    p.add_argument("--damage_consecutive", type=int, default=1)
    p.add_argument("--energy_drop_eV", type=float, default=1.0)
    p.add_argument("--thickness_A", type=float, default=3.35)
    p.add_argument("--crack_length_A", type=float, default=10.0)
    p.add_argument("--crack_y_fraction", type=float, default=0.55)
    p.add_argument("--inclusion_radius_A", type=float, default=15.0)
    p.add_argument("--inclusion_center_x_fraction", type=float, default=0.50)
    p.add_argument("--inclusion_center_y_fraction", type=float, default=0.30)
    p.add_argument("--plane_condition", choices=["plane_stress", "plane_strain"], default="plane_stress")
    p.add_argument("--gc_method", choices=["work_to_peak", "work_to_fracture", "lefm", "2d_work_to_peak"], default="work_to_peak")
    p.add_argument("--l0_norm", type=float, default=0.015)
    p.add_argument("--l_A", type=float, default=None)
    p.add_argument("--reference_graphene_E_GPa", type=float, default=None)
    p.add_argument("--no_plots", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if args.elastic_max <= args.elastic_min:
        raise ValueError("elastic_max must be greater than elastic_min.")
    if not 0 < args.stress_drop_fraction < 1:
        raise ValueError("stress_drop_fraction must lie between 0 and 1.")

    source = Path(args.input)
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)

    df = read_table(source)
    props, processed = extract(df, args, source)

    processed.to_csv(folder/"processed_hard_inclusion_curve.csv", index=False)
    pd.DataFrame([props]).to_csv(folder/"effective_hard_inclusion_properties.csv", index=False)

    data = {args.size: {args.material_key: {k: json_safe(v) for k,v in xdem_entry(props).items()}}}
    (folder/"xdem_effective_hard_inclusion_params.json").write_text(
        json.dumps(data, indent=4, allow_nan=False), encoding="utf-8"
    )
    write_summary(folder/"hard_inclusion_summary.txt", props)
    if not args.no_plots:
        plots(processed, props, folder, args)

    print("\nExtracted effective graphene-hBN hard-inclusion properties")
    print(f"E (3D)           = {props['E_effective_GPa']:.8g} GPa")
    print(f"E (2D)           = {props['E_effective_2D_N_m']:.8g} N/m")
    print(f"Poisson ratio    = {props['nu_effective_yx']:.8g}")
    print(f"Peak stress      = {props['sigma_c_effective_GPa']:.8g} GPa")
    print(f"Peak strain      = {props['epsilon_c']:.8g}")
    print(f"Fracture onset   = {props['fracture_onset_strain']:.8g} ({props['fracture_onset_source']})")
    print(f"Gc work-to-peak  = {props['Gc_work_to_peak_J_m2']:.8g} J/m2")
    print(f"Kc edge estimate = {props['Kc_edge_MPa_sqrt_m']:.8g} MPa sqrt(m)")
    print(f"Gc LEFM estimate = {props['Gc_LEFM_J_m2']:.8g} J/m2")
    print(f"Selected XDEM Gc = {props['Gc_selected_J_m2']:.8g} J/m2 ({props['Gc_selected_method']})")
    print(f"\nOutputs: {folder}")
    print("IMPORTANT: these are effective composite properties; pristine graphene and hBN runs are needed for phase-resolved XDEM.")


if __name__ == "__main__":
    main()
