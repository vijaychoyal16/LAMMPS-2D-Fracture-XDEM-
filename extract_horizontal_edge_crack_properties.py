#!/usr/bin/env python3
"""
Extract mechanical/XDEM parameters from horizontal edge-crack LAMMPS top-pull files.

Geometry for the attached image:
    - Horizontal edge crack from left edge
    - Crack angle = 0 degree
    - Crack length a/Lx = 0.50 by default
    - Top boundary pulled in +y direction
    - Bottom boundary fixed

Main outputs:
    xdem_horizontal_edge_outputs/all_horizontal_edge_mechanical_properties.csv
    xdem_horizontal_edge_outputs/all_horizontal_edge_processed_stress_strain_curves.csv
    xdem_horizontal_edge_outputs/xdem_material_params_by_size_for_xdem.json
    xdem_horizontal_edge_outputs/xdem_material_params_selected.json
    xdem_horizontal_edge_outputs/xdem_horizontal_edge_summary_table.txt

Expected files, for example:
    Gr/stress_strain_y_0K_top_pull_corrected_data_100x100.dat
    100_100/Gr/stress_strain_y_0K_top_pull_corrected_data_100x100.dat
    stress_strain_y_0K_top_pull_corrected_data_100x100.dat
"""

from pathlib import Path
import re
import json
import numpy as np
import pandas as pd

# ============================================================
# User settings
# ============================================================

ROOT_DIR = Path(".")
OUTPUT_DIR = Path("xdem_horizontal_edge_outputs")

MATERIALS = ["Gr"]
SIZES = ["100_100", "200_200", "300_300", "400_400", "500_500"]  # set None to auto-use all sizes found

THICKNESS_A = {
    "Gr": 3.35,
    "graphene": 3.35,
    "BN": 3.35,
    "BC3": 3.35,
    "BC6N": 3.35,
    "C3N": 3.35,
    "BCN": 3.35,
    "Bi2Se3": 6.91,
}

DEFAULT_NU = {
    "Gr": 0.16,
    "graphene": 0.16,
    "BN": 0.22,
    "BC3": 0.18,
    "BC6N": 0.20,
    "C3N": 0.22,
    "BCN": 0.20,
}

ELASTIC_MIN = 0.0
ELASTIC_MAX = 0.02
L0_NORM = 0.015
EDGE_CRACK_LENGTH_NORM = 0.50
SELECTION_MODE = "largest"  # largest or average_largest2

GLOB_PATTERNS = [
    "**/stress_strain_y_0K_top_pull_corrected*.dat",
    "**/stress_strain_y_0K_top_pull*.dat",
]

SIZE_REGEX = re.compile(r"(?P<nx>\d+(?:\.\d+)?)\s*[xX_]\s*(?P<ny>\d+(?:\.\d+)?)")


def integrate_trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def normalize_size(size_text):
    s = str(size_text).strip().replace(" ", "").replace("X", "x")
    if "x" in s:
        a, b = s.split("x", 1)
    elif "_" in s:
        a, b = s.split("_", 1)
    else:
        return s
    return f"{int(float(a))}_{int(float(b))}"


def size_to_number(size_name):
    s = normalize_size(size_name)
    nums = []
    for p in s.split("_"):
        try:
            nums.append(float(p))
        except ValueError:
            pass
    return max(nums) if nums else np.nan


def parse_size_from_path(path):
    path = Path(path)
    m = SIZE_REGEX.search(path.name)
    if m:
        return normalize_size(f"{m.group('nx')}_{m.group('ny')}")
    for parent in path.parents:
        m = SIZE_REGEX.search(parent.name)
        if m:
            return normalize_size(f"{m.group('nx')}_{m.group('ny')}")
    return "unknown"


def infer_material_from_path(path):
    parts = Path(path).parts
    for mat in MATERIALS:
        if mat in parts:
            return mat
    return MATERIALS[0]


def material_edge_key(material):
    return f"{material}_edge_horizontal"


def discover_files():
    found_paths = []
    for pattern in GLOB_PATTERNS:
        found_paths.extend(ROOT_DIR.glob(pattern))

    unique = {}
    for f in found_paths:
        if not f.is_file():
            continue
        if OUTPUT_DIR.name in f.parts:
            continue
        unique[str(f.resolve())] = f

    records = []
    for f in unique.values():
        material = infer_material_from_path(f)
        size = parse_size_from_path(f)
        if SIZES is not None:
            size_set = {normalize_size(s) for s in SIZES}
            if size not in size_set:
                continue
        records.append({"material": material, "size": size, "file": f})

    return sorted(records, key=lambda r: (r["material"], size_to_number(r["size"]), str(r["file"])))


def pick_strain_column(df):
    priority = ["strain_y", "epsilon_y", "eps_y", "strain", "engineering_strain"]
    for c in priority:
        if c in df.columns:
            return c
    raise ValueError(f"No recognized strain column found. Columns: {df.columns.tolist()}")


def pick_stress_column(df, require=True):
    priority = [
        "reaction_syy_GPa_corrected",
        "reaction_syy_GPa",
        "syy_GPa_corrected",
        "syy_GPa",
        "virial_syy_GPa",
        "pyy_GPa",
    ]
    for c in priority:
        if c in df.columns:
            return c
    if require:
        raise ValueError(f"No recognized y-stress column found. Columns: {df.columns.tolist()}")
    return None


def read_lammps_table(filename):
    df = pd.read_csv(filename, sep=r"\s+", engine="python", comment="#")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    strain_col = pick_strain_column(df)
    stress_col = pick_stress_column(df, require=False)
    if stress_col is None:
        raise ValueError(f"{filename} has no recognized y-stress column. Columns: {df.columns.tolist()}")

    df = df.dropna(subset=[strain_col, stress_col]).copy()
    df = df.sort_values(strain_col).drop_duplicates(subset=[strain_col], keep="last").reset_index(drop=True)
    if strain_col != "strain_y":
        df["strain_y"] = df[strain_col]
    return df


def robust_poisson_ratio(df, material):
    if "strain_x" not in df.columns:
        return DEFAULT_NU.get(material, 0.20), "default_no_strain_x"

    elastic = df[
        (df["strain_y"] >= ELASTIC_MIN)
        & (df["strain_y"] <= ELASTIC_MAX)
        & (df["strain_y"].abs() > 1.0e-12)
    ].copy()

    if len(elastic) < 3:
        return DEFAULT_NU.get(material, 0.20), "default_too_few_points"

    try:
        slope_xy, _ = np.polyfit(elastic["strain_y"].to_numpy(float), elastic["strain_x"].to_numpy(float), 1)
        nu = -float(slope_xy)
        if not np.isfinite(nu) or nu < 0.0 or nu > 0.7:
            return DEFAULT_NU.get(material, 0.20), "default_unphysical_fit"
        return nu, "fit_strain_x"
    except Exception:
        return DEFAULT_NU.get(material, 0.20), "default_fit_failed"


def extract_properties(df, material, size_name, filename):
    stress_col = pick_stress_column(df)
    strain = df["strain_y"].to_numpy(float)
    stress_raw = df[stress_col].to_numpy(float)

    stress_corr = stress_raw - stress_raw[0]
    if abs(np.nanmin(stress_corr)) > abs(np.nanmax(stress_corr)):
        stress_corr = -stress_corr
        stress_raw = -stress_raw

    df["main_stress_column"] = stress_col
    df["main_stress_raw_GPa"] = stress_raw
    df["main_stress_corrected_GPa"] = stress_corr

    if "PE_per_atom_eV" in df.columns:
        df["dPE_per_atom_eV"] = df["PE_per_atom_eV"] - df["PE_per_atom_eV"].iloc[0]
    elif "dPE_per_atom_eV" not in df.columns:
        df["dPE_per_atom_eV"] = np.nan

    elastic = df[(df["strain_y"] >= ELASTIC_MIN) & (df["strain_y"] <= ELASTIC_MAX)].copy()
    if len(elastic) < 3:
        elastic = df.head(min(6, len(df))).copy()
    if len(elastic) < 2:
        raise ValueError(f"Not enough points for elastic fit: {filename}")

    Ey_GPa, intercept = np.polyfit(
        elastic["strain_y"].to_numpy(float),
        elastic["main_stress_corrected_GPa"].to_numpy(float),
        1,
    )

    nu_yx, nu_source = robust_poisson_ratio(df, material)

    idx_peak = int(np.nanargmax(stress_corr))
    sigma_c_GPa = float(stress_corr[idx_peak])
    epsilon_c = float(strain[idx_peak])

    sub_strain = strain[:idx_peak + 1]
    sub_stress = np.maximum(stress_corr[:idx_peak + 1], 0.0)
    toughness_GPa = float(integrate_trapezoid(sub_stress, sub_strain)) if len(sub_strain) >= 2 else np.nan
    toughness_MJ_m3 = toughness_GPa * 1000.0 if np.isfinite(toughness_GPa) else np.nan

    h_A = float(THICKNESS_A.get(material, 3.35))
    Gc_initial_J_m2 = toughness_GPa * h_A * 0.1 if np.isfinite(toughness_GPa) else np.nan

    natoms_initial = int(df["natoms"].iloc[0]) if "natoms" in df.columns and pd.notna(df["natoms"].iloc[0]) else -1
    natoms_final = int(df["natoms"].iloc[-1]) if "natoms" in df.columns and pd.notna(df["natoms"].iloc[-1]) else -1

    Lx_initial = float(df["Lx_A"].iloc[0]) if "Lx_A" in df.columns else np.nan
    Ly_initial = float(df["Ly_A"].iloc[0]) if "Ly_A" in df.columns else np.nan
    Lz_initial = float(df["Lz_A"].iloc[0]) if "Lz_A" in df.columns else np.nan
    width_A = float(df["width_A"].iloc[0]) if "width_A" in df.columns else Lx_initial
    gauge_A = float(df["gauge_A"].iloc[0]) if "gauge_A" in df.columns else Ly_initial

    size_norm = normalize_size(size_name)
    size_numeric = size_to_number(size_norm)

    props = {
        "material": material,
        "material_edge_key": material_edge_key(material),
        "geometry": "horizontal_edge_crack",
        "fracture_mode": "mode_I_opening",
        "loading_direction": "y",
        "crack_angle_deg": 0.0,
        "edge_crack_length_norm": float(EDGE_CRACK_LENGTH_NORM),
        "size": size_norm,
        "size_A": float(size_numeric),
        "file": str(filename),
        "main_stress_column": stress_col,
        "n_points": int(len(df)),
        "natoms_initial": natoms_initial,
        "natoms_final": natoms_final,
        "Lx_A_initial": Lx_initial,
        "Ly_A_initial": Ly_initial,
        "Lz_A": Lz_initial,
        "width_A": width_A,
        "gauge_A": gauge_A,
        "thickness_A": h_A,
        "elastic_min": float(ELASTIC_MIN),
        "elastic_max": float(ELASTIC_MAX),
        "residual_stress_GPa": float(stress_raw[0]),
        "Ey_corrected_GPa": float(Ey_GPa),
        "Ey_MPa": float(Ey_GPa * 1000.0),
        "elastic_intercept_GPa": float(intercept),
        "nu_yx": float(nu_yx),
        "nu_source": nu_source,
        "sigma_c_GPa": sigma_c_GPa,
        "epsilon_c": epsilon_c,
        "toughness_integral_GPa": toughness_GPa,
        "toughness_MJ_m3": toughness_MJ_m3,
        "Gc_initial_J_m2": float(Gc_initial_J_m2),
        "l0_norm": float(L0_NORM),
    }

    df["material"] = material
    df["material_edge_key"] = material_edge_key(material)
    df["geometry"] = "horizontal_edge_crack"
    df["fracture_mode"] = "mode_I_opening"
    df["loading_direction"] = "y"
    df["crack_angle_deg"] = 0.0
    df["edge_crack_length_norm"] = float(EDGE_CRACK_LENGTH_NORM)
    df["size"] = size_norm
    df["size_A"] = float(size_numeric)
    df["source_file"] = str(filename)

    return props, df


def row_to_xdem_params(row):
    return {
        "selected_size": str(row["size"]),
        "material": str(row["material"]),
        "geometry": "horizontal_edge_crack",
        "fracture_mode": "mode_I_opening",
        "loading_direction": "y",
        "crack_angle_deg": 0.0,
        "edge_crack_length_norm": float(row["edge_crack_length_norm"]),
        "material_key": str(row["material_edge_key"]),
        "E_GPa": float(row["Ey_corrected_GPa"]),
        "E_MPa": float(row["Ey_MPa"]),
        "nu": float(row["nu_yx"]),
        "nu_source": str(row["nu_source"]),
        "sigma_c_GPa": float(row["sigma_c_GPa"]),
        "epsilon_c": float(row["epsilon_c"]),
        "Gc_initial_J_m2": float(row["Gc_initial_J_m2"]),
        "thickness_A": float(row["thickness_A"]),
        "l0_norm": float(row["l0_norm"]),
        "domain_L_norm": 1.0,
        "domain_W_norm": 1.0,
        "suggested_xdem_material_key": str(row["material_edge_key"]),
        "note": (
            "Horizontal left-edge crack under y-direction top-pull loading. "
            "E and peak response are extracted from reaction_syy_GPa_corrected "
            "or the best available y-stress column. Gc_initial_J_m2 is an approximate "
            "toughness-based estimate and should be calibrated with gc_scale."
        ),
    }


def select_summary(summary_df):
    selected = {}
    for material, sub in summary_df.groupby("material"):
        sub = sub.sort_values("size_A").reset_index(drop=True)
        if SELECTION_MODE == "largest" or len(sub) < 2:
            row = sub.iloc[-1].copy()
        elif SELECTION_MODE == "average_largest2":
            avg = sub.tail(2).copy()
            row = avg.iloc[-1].copy()
            for c in ["Ey_corrected_GPa", "Ey_MPa", "nu_yx", "sigma_c_GPa", "epsilon_c", "Gc_initial_J_m2", "thickness_A", "l0_norm"]:
                row[c] = avg[c].mean()
            row["size"] = "average_largest2"
            row["size_A"] = avg["size_A"].mean()
        else:
            raise ValueError(f"Unknown SELECTION_MODE: {SELECTION_MODE}")
        selected[material_edge_key(material)] = row_to_xdem_params(row)
    return selected


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = discover_files()

    if not records:
        raise RuntimeError(
            "No matching files found. Expected names like:\n"
            "  Gr/stress_strain_y_0K_top_pull_corrected_data_100x100.dat\n"
            "  100_100/Gr/stress_strain_y_0K_top_pull_corrected_data_100x100.dat\n"
            "Check ROOT_DIR, MATERIALS, SIZES, and GLOB_PATTERNS."
        )

    print("Found files:")
    for r in records:
        print(f"  material={r['material']:6s} size={r['size']:8s} file={r['file']}")

    all_props = []
    all_curves = []
    failed = []

    for r in records:
        try:
            df = read_lammps_table(r["file"])
            props, curve = extract_properties(df, r["material"], r["size"], r["file"])
            all_props.append(props)
            all_curves.append(curve)
            print(
                f"OK {props['material']:6s} {props['size']:8s}: "
                f"E={props['Ey_corrected_GPa']:.3f} GPa, "
                f"nu={props['nu_yx']:.3f} ({props['nu_source']}), "
                f"sigma_c={props['sigma_c_GPa']:.3f} GPa, "
                f"eps_c={props['epsilon_c']:.4f}, "
                f"Gc={props['Gc_initial_J_m2']:.4g} J/m2"
            )
        except Exception as exc:
            failed.append((str(r["file"]), str(exc)))
            print(f"ERROR processing {r['file']}: {exc}")

    if not all_props:
        raise RuntimeError("No valid files processed.")

    summary_df = pd.DataFrame(all_props).sort_values(["material", "size_A"]).reset_index(drop=True)
    curves_df = pd.concat(all_curves, ignore_index=True)

    summary_csv = OUTPUT_DIR / "all_horizontal_edge_mechanical_properties.csv"
    curves_csv = OUTPUT_DIR / "all_horizontal_edge_processed_stress_strain_curves.csv"
    summary_df.to_csv(summary_csv, index=False)
    curves_df.to_csv(curves_csv, index=False)

    by_size_for_xdem = {}
    for _, row in summary_df.iterrows():
        size = str(row["size"])
        key = str(row["material_edge_key"])
        by_size_for_xdem.setdefault(size, {})
        by_size_for_xdem[size][key] = row_to_xdem_params(row)

    json_for_xdem = OUTPUT_DIR / "xdem_material_params_by_size_for_xdem.json"
    with open(json_for_xdem, "w") as f:
        json.dump(by_size_for_xdem, f, indent=4)

    selected = select_summary(summary_df)
    json_selected = OUTPUT_DIR / "xdem_material_params_selected.json"
    with open(json_selected, "w") as f:
        json.dump(selected, f, indent=4)

    txt_file = OUTPUT_DIR / "xdem_horizontal_edge_summary_table.txt"
    with open(txt_file, "w") as f:
        f.write("Horizontal edge-crack y-direction top-pull mechanical properties for XDEM\n")
        f.write("==========================================================================\n")
        f.write("Geometry: left horizontal edge crack, crack_angle_deg = 0.0\n")
        f.write(f"Edge crack length norm: {EDGE_CRACK_LENGTH_NORM}\n")
        f.write(f"Elastic fitting range: {ELASTIC_MIN} to {ELASTIC_MAX}\n")
        f.write(f"Selection mode: {SELECTION_MODE}\n\n")
        cols = [
            "material", "size", "Ey_corrected_GPa", "nu_yx", "nu_source",
            "sigma_c_GPa", "epsilon_c", "Gc_initial_J_m2", "main_stress_column"
        ]
        f.write(summary_df[cols].to_string(index=False))
        f.write("\n\n")
        if failed:
            f.write("Failed files:\n")
            for path, err in failed:
                f.write(f"  {path}: {err}\n")

    print("\nDone.")
    print(f"Summary CSV:          {summary_csv}")
    print(f"Processed curves CSV: {curves_csv}")
    print(f"XDEM JSON by size:    {json_for_xdem}")
    print(f"Selected JSON:        {json_selected}")
    print(f"Text summary:         {txt_file}")
    print("\nUse this JSON for XDEM:")
    print(f"  {json_for_xdem}")
    print("\nExample:")
    print("  --params xdem_horizontal_edge_outputs/xdem_material_params_by_size_for_xdem.json")
    print("  --size 300_300")
    print("  --material Gr_edge_horizontal")


if __name__ == "__main__":
    main()
