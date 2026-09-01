#!/usr/bin/env python3
"""
Generate single-layer graphene sheets containing coherent circular h-BN inclusions.

Inputs
------
1. A graphene POSCAR containing carbon atoms.
2. An h-BN POSCAR containing B and N atoms. The BN file is used to estimate the
   natural B-N/graphene C-C bond-length mismatch. The generated composite is
   coherent with the graphene lattice: C sites in the inclusion are converted
   into alternating B and N sublattices.

Outputs for every requested nominal size
----------------------------------------
* Gr_BN_inclusion_<size>.data : graphene matrix + circular BN inclusion
* graphene_<size>.data        : pure graphene reference
* hBN_coherent_<size>.data    : pure BN constrained to the graphene lattice
* Gr_BN_inclusion_<size>.xyz  : structure for quick visualization
* generation_summary.csv      : actual cell dimensions and atom counts

LAMMPS atom-type mapping is fixed:
    type 1 = C
    type 2 = B
    type 3 = N

The LAMMPS data files use atom_style atomic.

Example
-------
python make_graphene_bn_inclusion.py \
    --graphene_poscar POSCAR_graphene \
    --bn_poscar POSCAR_BN \
    --sizes 100 200 500 \
    --radius_fraction 0.15 \
    --vacuum 25.0 \
    --outdir graphene_BN_inclusions
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


MASS = {"C": 12.011, "B": 10.810, "N": 14.007}
TYPE_ID = {"C": 1, "B": 2, "N": 3}


@dataclass
class Structure:
    cell: np.ndarray          # shape (3, 3), lattice vectors stored as rows
    positions: np.ndarray     # Cartesian positions, shape (N, 3)
    symbols: List[str]


def read_poscar(filename: str | Path) -> Structure:
    """Read a VASP POSCAR/CONTCAR without external dependencies."""
    filename = Path(filename)
    raw = [line.rstrip() for line in filename.read_text().splitlines() if line.strip()]
    if len(raw) < 8:
        raise ValueError(f"{filename} does not look like a valid POSCAR.")

    scale = float(raw[1].split()[0])
    cell = np.array([[float(x) for x in raw[i].split()[:3]] for i in range(2, 5)])

    # VASP allows a negative scale to specify a target cell volume.
    if scale < 0.0:
        target_volume = abs(scale)
        current_volume = abs(np.linalg.det(cell))
        if current_volume <= 0.0:
            raise ValueError(f"Invalid cell volume in {filename}.")
        cell *= (target_volume / current_volume) ** (1.0 / 3.0)
    else:
        cell *= scale

    tokens_5 = raw[5].split()
    has_symbols = not all(_is_integer(tok) for tok in tokens_5)

    if has_symbols:
        species = tokens_5
        counts = [int(x) for x in raw[6].split()]
        line_index = 7
    else:
        raise ValueError(
            f"{filename} uses an old-style POSCAR without element names. "
            "Please add the element-symbol line, e.g. 'C' or 'B N'."
        )

    if len(species) != len(counts):
        raise ValueError(f"Element and count lines are inconsistent in {filename}.")

    if raw[line_index].strip().lower().startswith("s"):
        line_index += 1

    coordinate_mode = raw[line_index].strip().lower()
    line_index += 1
    natoms = sum(counts)

    if len(raw) < line_index + natoms:
        raise ValueError(f"Not enough atomic-coordinate lines in {filename}.")

    coordinates = np.array(
        [[float(x) for x in raw[line_index + i].split()[:3]] for i in range(natoms)],
        dtype=float,
    )

    if coordinate_mode.startswith("d"):
        positions = coordinates @ cell
    elif coordinate_mode.startswith("c") or coordinate_mode.startswith("k"):
        # Cartesian coordinates are multiplied by the positive POSCAR scale.
        positions = coordinates * (scale if scale > 0.0 else 1.0)
    else:
        raise ValueError(f"Unknown coordinate mode '{raw[line_index - 1]}' in {filename}.")

    symbols: List[str] = []
    for element, count in zip(species, counts):
        symbols.extend([element] * count)

    return Structure(cell=cell, positions=positions, symbols=symbols)


def _is_integer(text: str) -> bool:
    try:
        int(text)
        return True
    except ValueError:
        return False


def inplane_angle_degrees(cell: np.ndarray) -> float:
    a = cell[0, :2]
    b = cell[1, :2]
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def transform_supercell(structure: Structure, transform: np.ndarray) -> Structure:
    """
    Apply a small integer supercell transformation.

    The implementation uses a bounded translation search and fractional-coordinate
    filtering, which is robust for the 2D determinant-2 transformations used here.
    """
    transform = np.asarray(transform, dtype=int)
    if transform.shape != (3, 3):
        raise ValueError("Supercell transformation must have shape (3, 3).")

    new_cell = transform @ structure.cell
    inv_new_cell = np.linalg.inv(new_cell)
    determinant = int(round(abs(np.linalg.det(transform))))
    expected_atoms = determinant * len(structure.symbols)

    search = int(np.max(np.abs(transform))) + 3
    candidates: List[Tuple[np.ndarray, str]] = []

    for ia in range(-search, search + 1):
        for ib in range(-search, search + 1):
            # Do not replicate through the vacuum direction.
            translation = np.array([ia, ib, 0], dtype=float) @ structure.cell
            for position, symbol in zip(structure.positions, structure.symbols):
                cart = position + translation
                frac = cart @ inv_new_cell
                if (
                    -1.0e-8 <= frac[0] < 1.0 - 1.0e-8
                    and -1.0e-8 <= frac[1] < 1.0 - 1.0e-8
                    and -1.0e-8 <= frac[2] < 1.0 - 1.0e-8
                ):
                    frac = frac - np.floor(frac + 1.0e-10)
                    candidates.append((frac, symbol))

    # Remove duplicate boundary images.
    unique: dict[Tuple[int, int, int, str], Tuple[np.ndarray, str]] = {}
    for frac, symbol in candidates:
        key = (
            int(round(frac[0] * 1.0e9)),
            int(round(frac[1] * 1.0e9)),
            int(round(frac[2] * 1.0e9)),
            symbol,
        )
        unique[key] = (frac, symbol)

    values = list(unique.values())
    if len(values) != expected_atoms:
        raise RuntimeError(
            "Could not construct the rectangular supercell reliably: "
            f"expected {expected_atoms} atoms but found {len(values)}. "
            "Use a rectangular input POSCAR or inspect the lattice vectors."
        )

    fractions = np.array([item[0] for item in values])
    symbols = [item[1] for item in values]
    positions = fractions @ new_cell
    return Structure(cell=new_cell, positions=positions, symbols=symbols)


def make_rectangular_2d(structure: Structure, angle_tolerance: float = 3.0) -> Structure:
    """Convert a common hexagonal primitive cell to a rectangular 2D cell."""
    angle = inplane_angle_degrees(structure.cell)

    if abs(angle - 90.0) <= angle_tolerance:
        rectangular = structure
    elif abs(angle - 60.0) <= angle_tolerance:
        transform = np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 1]], dtype=int)
        rectangular = transform_supercell(structure, transform)
    elif abs(angle - 120.0) <= angle_tolerance:
        transform = np.array([[1, 1, 0], [1, -1, 0], [0, 0, 1]], dtype=int)
        rectangular = transform_supercell(structure, transform)
    else:
        raise ValueError(
            f"Unsupported in-plane lattice angle {angle:.3f} degrees. "
            "Provide either a rectangular cell or a standard 60/120-degree hexagonal cell."
        )

    a = rectangular.cell[0]
    b = rectangular.cell[1]
    e1 = a / np.linalg.norm(a)
    b_perp = b - np.dot(b, e1) * e1
    if np.linalg.norm(b_perp) < 1.0e-10:
        raise ValueError("The first two lattice vectors are linearly dependent.")
    e2 = b_perp / np.linalg.norm(b_perp)
    e3 = np.cross(e1, e2)
    e3 /= np.linalg.norm(e3)

    # Preserve a right-handed orientation consistent with the original c vector.
    if np.dot(rectangular.cell[2], e3) < 0.0:
        e2 *= -1.0
        e3 = np.cross(e1, e2)
        e3 /= np.linalg.norm(e3)

    basis = np.column_stack((e1, e2, e3))
    rotated_positions = rectangular.positions @ basis

    lx = np.linalg.norm(a)
    ly = np.linalg.norm(b_perp)
    lz = abs(np.dot(rectangular.cell[2], e3))
    if lz < 1.0e-8:
        lz = np.linalg.norm(rectangular.cell[2])

    rotated_positions[:, 0] %= lx
    rotated_positions[:, 1] %= ly
    rotated_positions[:, 2] %= lz

    order = np.lexsort(
        (
            np.round(rotated_positions[:, 0], 10),
            np.round(rotated_positions[:, 1], 10),
            np.round(rotated_positions[:, 2], 10),
        )
    )
    rotated_positions = rotated_positions[order]
    symbols = [rectangular.symbols[i] for i in order]

    return Structure(
        cell=np.diag([lx, ly, lz]),
        positions=rotated_positions,
        symbols=symbols,
    )


def minimum_periodic_distance(structure: Structure, allowed_pairs: set[frozenset[str]]) -> float:
    """Return the shortest allowed in-plane bonded distance using periodic images."""
    positions = structure.positions
    lx, ly = structure.cell[0, 0], structure.cell[1, 1]
    best = float("inf")

    for i, (ri, si) in enumerate(zip(positions, structure.symbols)):
        for j, (rj, sj) in enumerate(zip(positions, structure.symbols)):
            if frozenset((si, sj)) not in allowed_pairs:
                continue
            for sx in (-1, 0, 1):
                for sy in (-1, 0, 1):
                    if i == j and sx == 0 and sy == 0:
                        continue
                    displacement = rj + np.array([sx * lx, sy * ly, 0.0]) - ri
                    distance = float(np.linalg.norm(displacement))
                    if 1.0e-8 < distance < best:
                        best = distance

    if not np.isfinite(best):
        raise RuntimeError("Could not determine a nearest-neighbour distance.")
    return best


def graphene_sublattice_labels(base: Structure) -> np.ndarray:
    """Bipartite-color the graphene honeycomb lattice in the rectangular base cell."""
    if set(base.symbols) != {"C"}:
        raise ValueError("The graphene POSCAR must contain only carbon atoms.")

    d_cc = minimum_periodic_distance(base, {frozenset(("C", "C"))})
    cutoff = 1.20 * d_cc
    n = len(base.symbols)
    adjacency: List[set[int]] = [set() for _ in range(n)]
    lx, ly = base.cell[0, 0], base.cell[1, 1]

    for i, ri in enumerate(base.positions):
        for j, rj in enumerate(base.positions):
            for sx in (-1, 0, 1):
                for sy in (-1, 0, 1):
                    if i == j and sx == 0 and sy == 0:
                        continue
                    displacement = rj + np.array([sx * lx, sy * ly, 0.0]) - ri
                    distance = float(np.linalg.norm(displacement))
                    if distance <= cutoff:
                        adjacency[i].add(j)

    colors = -np.ones(n, dtype=int)
    for start in range(n):
        if colors[start] != -1:
            continue
        colors[start] = 0
        queue = [start]
        while queue:
            i = queue.pop(0)
            for j in adjacency[i]:
                if colors[j] == -1:
                    colors[j] = 1 - colors[i]
                    queue.append(j)
                elif colors[j] == colors[i]:
                    raise RuntimeError(
                        "Graphene lattice could not be bipartite-colored. "
                        "Check that the POSCAR is a clean honeycomb graphene cell."
                    )

    return colors


def build_sheet(
    base: Structure,
    base_colors: np.ndarray,
    target_x: float,
    target_y: float,
    vacuum: float,
) -> Tuple[np.ndarray, np.ndarray, float, float, int, int]:
    """Repeat a rectangular graphene cell to the closest commensurate dimensions."""
    ax = float(base.cell[0, 0])
    ay = float(base.cell[1, 1])
    nx = max(1, int(round(target_x / ax)))
    ny = max(1, int(round(target_y / ay)))
    lx = nx * ax
    ly = ny * ay

    layer_z = base.positions[:, 2] - np.mean(base.positions[:, 2])
    positions: List[np.ndarray] = []
    colors: List[int] = []

    for ix in range(nx):
        for iy in range(ny):
            shift = np.array([ix * ax, iy * ay, vacuum / 2.0])
            local = base.positions.copy()
            local[:, 2] = layer_z
            positions.extend(local + shift)
            colors.extend(base_colors.tolist())

    return (
        np.asarray(positions, dtype=float),
        np.asarray(colors, dtype=int),
        lx,
        ly,
        nx,
        ny,
    )


def inclusion_symbols(
    positions: np.ndarray,
    colors: np.ndarray,
    lx: float,
    ly: float,
    radius: float,
    center_fraction: Tuple[float, float],
    swap_bn: bool,
) -> Tuple[List[str], np.ndarray]:
    """Convert graphene sites inside a circular region to alternating B/N sites."""
    cx = center_fraction[0] * lx
    cy = center_fraction[1] * ly

    dx = positions[:, 0] - cx
    dy = positions[:, 1] - cy
    # Minimum-image convention makes the implementation safe near periodic edges.
    dx -= lx * np.rint(dx / lx)
    dy -= ly * np.rint(dy / ly)
    mask = dx * dx + dy * dy <= radius * radius

    symbols = np.full(len(positions), "C", dtype="<U1")
    if swap_bn:
        symbols[mask & (colors == 0)] = "N"
        symbols[mask & (colors == 1)] = "B"
    else:
        symbols[mask & (colors == 0)] = "B"
        symbols[mask & (colors == 1)] = "N"

    return symbols.tolist(), mask


def coherent_bn_symbols(colors: np.ndarray, swap_bn: bool) -> List[str]:
    symbols = np.empty(len(colors), dtype="<U1")
    if swap_bn:
        symbols[colors == 0] = "N"
        symbols[colors == 1] = "B"
    else:
        symbols[colors == 0] = "B"
        symbols[colors == 1] = "N"
    return symbols.tolist()


def write_lammps_data(
    filename: str | Path,
    positions: np.ndarray,
    symbols: Sequence[str],
    lx: float,
    ly: float,
    lz: float,
    comment: str,
) -> None:
    filename = Path(filename)
    present_types = sorted({TYPE_ID[s] for s in symbols})
    ntypes = max(present_types)

    with filename.open("w") as handle:
        handle.write(f"{comment}\n\n")
        handle.write(f"{len(symbols)} atoms\n")
        handle.write(f"{ntypes} atom types\n\n")
        handle.write(f"0.0 {lx:.12f} xlo xhi\n")
        handle.write(f"0.0 {ly:.12f} ylo yhi\n")
        handle.write(f"0.0 {lz:.12f} zlo zhi\n\n")

        handle.write("Masses\n\n")
        for element in ("C", "B", "N"):
            type_id = TYPE_ID[element]
            if type_id <= ntypes:
                handle.write(f"{type_id} {MASS[element]:.8f} # {element}\n")

        handle.write("\nAtoms # atomic\n\n")
        for atom_id, (position, symbol) in enumerate(zip(positions, symbols), start=1):
            handle.write(
                f"{atom_id} {TYPE_ID[symbol]} "
                f"{position[0]:.12f} {position[1]:.12f} {position[2]:.12f}\n"
            )


def write_xyz(filename: str | Path, positions: np.ndarray, symbols: Sequence[str]) -> None:
    filename = Path(filename)
    with filename.open("w") as handle:
        handle.write(f"{len(symbols)}\n")
        handle.write("Graphene matrix with coherent circular h-BN inclusion\n")
        for symbol, position in zip(symbols, positions):
            handle.write(
                f"{symbol:2s} {position[0]:.10f} {position[1]:.10f} {position[2]:.10f}\n"
            )


def parse_sizes(values: Iterable[str]) -> List[Tuple[float, float, str]]:
    """Accept values such as 100, 200, 100x150, or 100X150."""
    parsed: List[Tuple[float, float, str]] = []
    for value in values:
        clean = value.lower().replace("å", "").replace("a", "")
        if "x" in clean:
            first, second = clean.split("x", maxsplit=1)
            sx, sy = float(first), float(second)
            label = f"{_format_size(sx)}x{_format_size(sy)}"
        else:
            sx = sy = float(clean)
            label = f"{_format_size(sx)}x{_format_size(sy)}"
        if sx <= 0.0 or sy <= 0.0:
            raise ValueError("All requested dimensions must be positive.")
        parsed.append((sx, sy, label))
    return parsed


def _format_size(value: float) -> str:
    return str(int(value)) if abs(value - round(value)) < 1.0e-10 else str(value).replace(".", "p")


def count_symbols(symbols: Sequence[str]) -> dict[str, int]:
    return {element: int(sum(symbol == element for symbol in symbols)) for element in ("C", "B", "N")}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create graphene sheets with coherent circular h-BN inclusions."
    )
    parser.add_argument("--graphene_poscar", required=True, help="Graphene POSCAR/CONTCAR")
    parser.add_argument("--bn_poscar", required=True, help="h-BN POSCAR/CONTCAR")
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["100", "200", "500"],
        help="Nominal dimensions in Angstrom, e.g. 100 200 500 or 100x150",
    )
    parser.add_argument(
        "--radius_fraction",
        type=float,
        default=0.15,
        help="Inclusion radius divided by min(actual Lx, actual Ly); default 0.15",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Fixed inclusion radius in Angstrom. Overrides --radius_fraction.",
    )
    parser.add_argument(
        "--center_x",
        type=float,
        default=0.50,
        help="Inclusion-center x coordinate as a fraction of Lx; default 0.50",
    )
    parser.add_argument(
        "--center_y",
        type=float,
        default=0.50,
        help="Inclusion-center y coordinate as a fraction of Ly; default 0.50",
    )
    parser.add_argument("--vacuum", type=float, default=25.0, help="Cell height in Angstrom")
    parser.add_argument("--swap_bn", action="store_true", help="Exchange B and N sublattices")
    parser.add_argument(
        "--no_controls",
        action="store_true",
        help="Do not write pure-graphene and coherent-pure-BN reference files",
    )
    parser.add_argument("--outdir", default="graphene_BN_inclusions", help="Output directory")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    if args.vacuum <= 0.0:
        raise ValueError("--vacuum must be positive.")
    if args.radius is None and not (0.0 < args.radius_fraction < 0.5):
        raise ValueError("--radius_fraction must be between 0 and 0.5.")
    if not (0.0 <= args.center_x <= 1.0 and 0.0 <= args.center_y <= 1.0):
        raise ValueError("--center_x and --center_y must be between 0 and 1.")

    graphene_raw = read_poscar(args.graphene_poscar)
    bn_raw = read_poscar(args.bn_poscar)

    if set(graphene_raw.symbols) != {"C"}:
        raise ValueError("The graphene POSCAR must contain only C atoms.")
    if not set(bn_raw.symbols).issubset({"B", "N"}) or set(bn_raw.symbols) != {"B", "N"}:
        raise ValueError("The h-BN POSCAR must contain both B and N atoms and no other elements.")

    graphene_base = make_rectangular_2d(graphene_raw)
    bn_base = make_rectangular_2d(bn_raw)
    base_colors = graphene_sublattice_labels(graphene_base)

    d_cc = minimum_periodic_distance(graphene_base, {frozenset(("C", "C"))})
    d_bn = minimum_periodic_distance(bn_base, {frozenset(("B", "N"))})
    mismatch = 100.0 * (d_bn - d_cc) / d_cc

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    requested_sizes = parse_sizes(args.sizes)
    summary_rows = []

    print(f"Graphene nearest-neighbour C-C distance : {d_cc:.6f} Angstrom")
    print(f"h-BN nearest-neighbour B-N distance    : {d_bn:.6f} Angstrom")
    print(f"Natural BN/graphene bond mismatch       : {mismatch:+.3f} %")
    print("The generated BN inclusion is coherent and constrained to graphene lattice sites.\n")

    for target_x, target_y, label in requested_sizes:
        positions, colors, lx, ly, nx, ny = build_sheet(
            graphene_base,
            base_colors,
            target_x=target_x,
            target_y=target_y,
            vacuum=args.vacuum,
        )

        radius = args.radius if args.radius is not None else args.radius_fraction * min(lx, ly)
        if radius <= 0.0:
            raise ValueError("The inclusion radius must be positive.")
        if radius >= 0.5 * min(lx, ly):
            raise ValueError(
                f"Radius {radius:.3f} Angstrom is too large for cell {lx:.3f} x {ly:.3f} Angstrom."
            )

        composite_symbols, mask = inclusion_symbols(
            positions,
            colors,
            lx=lx,
            ly=ly,
            radius=radius,
            center_fraction=(args.center_x, args.center_y),
            swap_bn=args.swap_bn,
        )
        graphene_symbols = ["C"] * len(positions)
        bn_symbols = coherent_bn_symbols(colors, swap_bn=args.swap_bn)

        composite_file = outdir / f"Gr_BN_inclusion_{label}.data"
        xyz_file = outdir / f"Gr_BN_inclusion_{label}.xyz"
        write_lammps_data(
            composite_file,
            positions,
            composite_symbols,
            lx,
            ly,
            args.vacuum,
            comment=(
                "Graphene matrix with coherent circular h-BN inclusion; "
                "atom types: 1=C, 2=B, 3=N"
            ),
        )
        write_xyz(xyz_file, positions, composite_symbols)

        if not args.no_controls:
            write_lammps_data(
                outdir / f"graphene_{label}.data",
                positions,
                graphene_symbols,
                lx,
                ly,
                args.vacuum,
                comment="Pure graphene reference; atom type 1=C",
            )
            write_lammps_data(
                outdir / f"hBN_coherent_{label}.data",
                positions,
                bn_symbols,
                lx,
                ly,
                args.vacuum,
                comment="Coherent h-BN reference constrained to graphene lattice; types 2=B, 3=N",
            )

        counts = count_symbols(composite_symbols)
        summary_rows.append(
            {
                "requested_Lx_A": target_x,
                "requested_Ly_A": target_y,
                "actual_Lx_A": lx,
                "actual_Ly_A": ly,
                "vacuum_Lz_A": args.vacuum,
                "repeat_nx": nx,
                "repeat_ny": ny,
                "total_atoms": len(positions),
                "C_atoms": counts["C"],
                "B_atoms": counts["B"],
                "N_atoms": counts["N"],
                "BN_inclusion_atoms": int(np.count_nonzero(mask)),
                "inclusion_radius_A": radius,
                "center_x_fraction": args.center_x,
                "center_y_fraction": args.center_y,
                "CC_bond_A": d_cc,
                "BN_natural_bond_A": d_bn,
                "BN_graphene_mismatch_percent": mismatch,
                "data_file": composite_file.name,
            }
        )

        print(
            f"{label}: actual cell = {lx:.3f} x {ly:.3f} x {args.vacuum:.3f} A, "
            f"atoms = {len(positions)}, inclusion R = {radius:.3f} A, "
            f"C/B/N = {counts['C']}/{counts['B']}/{counts['N']}"
        )

    summary_file = outdir / "generation_summary.csv"
    with summary_file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nFiles written to: {outdir.resolve()}")
    print("LAMMPS type mapping: 1=C, 2=B, 3=N")


if __name__ == "__main__":
    main()
