#!/usr/bin/env python3
"""
Build rectangular graphene sheets with a coherent circular h-BN inclusion,
with and/or without a horizontal edge crack, and write LAMMPS data files.

The geometry follows a common fracture/inclusion specimen:

    Material 1: graphene rectangular matrix
    Material 2: circular h-BN inclusion
    Optional:   horizontal edge crack from the left or right boundary

The h-BN inclusion is generated coherently on graphene lattice sites by
converting the two graphene sublattices to B and N. This avoids overlapping
atoms and an artificial gap that would arise from independently cutting and
merging graphene and h-BN sheets.

LAMMPS atom-type mapping
------------------------
    type 1 = C
    type 2 = B
    type 3 = N

Required Python package
-----------------------
    numpy

Example: geometry matching W=100 A, H=200 A, R=15 A, a=10 A
----------------------------------------------------------------
python make_rectangular_graphene_bn_inclusion_crack.py \
    --graphene_poscar POSCAR_graphene \
    --bn_poscar POSCAR_BN \
    --sizes 100x200 \
    --variants both \
    --radius 15.0 \
    --center_x 0.50 \
    --center_y 0.30 \
    --crack_length 10.0 \
    --crack_y 110.0 \
    --crack_width 2.0 \
    --outdir Gr_BN_rectangular

Example: geometrically similar specimens of several sizes
----------------------------------------------------------
python make_rectangular_graphene_bn_inclusion_crack.py \
    --graphene_poscar POSCAR_graphene \
    --bn_poscar POSCAR_BN \
    --sizes 100x200 200x400 500x1000 \
    --variants both \
    --radius_fraction 0.15 \
    --center_x 0.50 \
    --center_y 0.30 \
    --crack_length_fraction 0.10 \
    --crack_y_fraction 0.55 \
    --crack_width 2.0 \
    --outdir Gr_BN_scaled

Outputs for each size
---------------------
    Gr_BN_<W>x<H>_no_crack.data
    Gr_BN_<W>x<H>_edge_crack.data
    corresponding .xyz and .vasp files
    Gr_BN_<W>x<H>_groups.lmp
    generation_summary.csv
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
    cell: np.ndarray       # lattice vectors stored as rows, shape (3, 3)
    positions: np.ndarray  # Cartesian coordinates, shape (N, 3)
    symbols: List[str]


def _is_integer(text: str) -> bool:
    try:
        int(text)
        return True
    except ValueError:
        return False


def read_poscar(filename: str | Path) -> Structure:
    """Read a VASP 5/6 POSCAR or CONTCAR without ASE/pymatgen."""
    filename = Path(filename)
    if not filename.is_file():
        raise FileNotFoundError(f"Input file not found: {filename}")

    raw = [line.rstrip() for line in filename.read_text().splitlines() if line.strip()]
    if len(raw) < 8:
        raise ValueError(f"{filename} does not look like a valid POSCAR.")

    scale = float(raw[1].split()[0])
    cell = np.array([[float(x) for x in raw[i].split()[:3]] for i in range(2, 5)])

    if scale < 0.0:
        target_volume = abs(scale)
        current_volume = abs(np.linalg.det(cell))
        if current_volume <= 0.0:
            raise ValueError(f"Invalid cell volume in {filename}.")
        cell *= (target_volume / current_volume) ** (1.0 / 3.0)
    else:
        cell *= scale

    tokens_5 = raw[5].split()
    if all(_is_integer(token) for token in tokens_5):
        raise ValueError(
            f"{filename} is an old-style POSCAR without element symbols. "
            "Add an element line, for example 'C' or 'B N'."
        )

    species = tokens_5
    counts = [int(x) for x in raw[6].split()]
    if len(species) != len(counts):
        raise ValueError(f"Element and count lines are inconsistent in {filename}.")

    line_index = 7
    if raw[line_index].lower().startswith("s"):
        line_index += 1

    coordinate_mode = raw[line_index].lower()
    line_index += 1
    natoms = sum(counts)

    if len(raw) < line_index + natoms:
        raise ValueError(f"Not enough coordinate lines in {filename}.")

    coordinates = np.array(
        [[float(x) for x in raw[line_index + i].split()[:3]] for i in range(natoms)],
        dtype=float,
    )

    if coordinate_mode.startswith("d"):
        positions = coordinates @ cell
    elif coordinate_mode.startswith("c") or coordinate_mode.startswith("k"):
        positions = coordinates * (scale if scale > 0.0 else 1.0)
    else:
        raise ValueError(f"Unknown coordinate mode in {filename}: {coordinate_mode}")

    symbols: List[str] = []
    for element, count in zip(species, counts):
        symbols.extend([element] * count)

    return Structure(cell=cell, positions=positions, symbols=symbols)


def inplane_angle_degrees(cell: np.ndarray) -> float:
    a = cell[0, :2]
    b = cell[1, :2]
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def transform_supercell(structure: Structure, transform: np.ndarray) -> Structure:
    """Apply a small integer supercell transform used for hexagonal-to-rectangular conversion."""
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
            "Could not construct the rectangular cell reliably: "
            f"expected {expected_atoms} atoms, found {len(values)}. "
            "Try supplying a rectangular input POSCAR."
        )

    fractions = np.array([item[0] for item in values])
    symbols = [item[1] for item in values]
    return Structure(new_cell, fractions @ new_cell, symbols)


def make_rectangular_2d(structure: Structure, angle_tolerance: float = 3.0) -> Structure:
    """Convert common 60/120-degree hexagonal cells to an orthogonal 2D cell."""
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
            f"Unsupported in-plane angle {angle:.3f} degrees. "
            "Use a rectangular cell or a standard 60/120-degree hexagonal cell."
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

    if np.dot(rectangular.cell[2], e3) < 0.0:
        e2 *= -1.0
        e3 = np.cross(e1, e2)
        e3 /= np.linalg.norm(e3)

    basis = np.column_stack((e1, e2, e3))
    positions = rectangular.positions @ basis

    lx = np.linalg.norm(a)
    ly = np.linalg.norm(b_perp)
    lz = abs(np.dot(rectangular.cell[2], e3))
    if lz < 1.0e-8:
        lz = np.linalg.norm(rectangular.cell[2])

    positions[:, 0] %= lx
    positions[:, 1] %= ly
    positions[:, 2] %= lz

    order = np.lexsort(
        (
            np.round(positions[:, 0], 10),
            np.round(positions[:, 1], 10),
            np.round(positions[:, 2], 10),
        )
    )
    return Structure(
        cell=np.diag([lx, ly, lz]),
        positions=positions[order],
        symbols=[rectangular.symbols[i] for i in order],
    )


def minimum_periodic_distance(
    structure: Structure,
    allowed_pairs: set[frozenset[str]],
) -> float:
    """Shortest allowed in-plane interatomic distance in a small periodic base cell."""
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
    """Assign graphene atoms to the two honeycomb sublattices."""
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
                    if float(np.linalg.norm(displacement)) <= cutoff:
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
                        "The graphene cell could not be bipartite-colored. "
                        "Check that it is an ideal honeycomb lattice."
                    )
    return colors


def build_sheet(
    base: Structure,
    base_colors: np.ndarray,
    target_x: float,
    target_y: float,
    vacuum: float,
) -> Tuple[np.ndarray, np.ndarray, float, float, int, int]:
    """Repeat the rectangular graphene base cell to near the requested W and H."""
    ax = float(base.cell[0, 0])
    ay = float(base.cell[1, 1])
    nx = max(1, int(round(target_x / ax)))
    ny = max(1, int(round(target_y / ay)))
    lx = nx * ax
    ly = ny * ay

    local = base.positions.copy()
    local[:, 2] -= np.mean(local[:, 2])

    positions: List[np.ndarray] = []
    colors: List[int] = []
    for ix in range(nx):
        for iy in range(ny):
            shift = np.array([ix * ax, iy * ay, vacuum / 2.0])
            positions.extend(local + shift)
            colors.extend(base_colors.tolist())

    positions_array = np.asarray(positions, dtype=float)
    colors_array = np.asarray(colors, dtype=int)

    # Deterministic order is useful for comparing cracked and uncracked models.
    order = np.lexsort((positions_array[:, 0], positions_array[:, 1], positions_array[:, 2]))
    return positions_array[order], colors_array[order], lx, ly, nx, ny


def assign_bn_inclusion(
    positions: np.ndarray,
    colors: np.ndarray,
    lx: float,
    ly: float,
    radius: float,
    center_fraction: Tuple[float, float],
    swap_bn: bool,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """Convert the two sublattices inside a circle from C to B and N."""
    cx = center_fraction[0] * lx
    cy = center_fraction[1] * ly
    distance2 = (positions[:, 0] - cx) ** 2 + (positions[:, 1] - cy) ** 2
    inclusion_mask = distance2 <= radius**2

    symbols = np.full(len(positions), "C", dtype="<U1")
    if swap_bn:
        symbols[inclusion_mask & (colors == 0)] = "N"
        symbols[inclusion_mask & (colors == 1)] = "B"
    else:
        symbols[inclusion_mask & (colors == 0)] = "B"
        symbols[inclusion_mask & (colors == 1)] = "N"

    return symbols, inclusion_mask, (cx, cy)


def edge_crack_keep_mask(
    positions: np.ndarray,
    lx: float,
    crack_length: float,
    crack_y: float,
    crack_width: float,
    side: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a horizontal edge crack by removing atoms within a capsule-shaped slit.

    The slit is the set of atoms within crack_width/2 of a horizontal line segment
    extending inward from the selected material edge. A rounded atomistic crack tip
    is therefore created automatically.
    """
    if side == "left":
        start = np.array([0.0, crack_y])
        end = np.array([crack_length, crack_y])
    elif side == "right":
        start = np.array([lx, crack_y])
        end = np.array([lx - crack_length, crack_y])
    else:
        raise ValueError("crack side must be 'left' or 'right'.")

    xy = positions[:, :2]
    segment = end - start
    segment_norm2 = float(np.dot(segment, segment))
    if segment_norm2 <= 0.0:
        raise ValueError("Crack length must be positive.")

    projection = ((xy - start) @ segment) / segment_norm2
    projection_clipped = np.clip(projection, 0.0, 1.0)
    closest = start + projection_clipped[:, None] * segment
    distance = np.linalg.norm(xy - closest, axis=1)

    removed = distance <= 0.5 * crack_width
    keep = ~removed
    return keep, removed


def write_lammps_data(
    filename: str | Path,
    positions: np.ndarray,
    symbols: Sequence[str],
    box_lx: float,
    box_ly: float,
    box_lz: float,
    comment: str,
) -> None:
    """Write an atom_style atomic LAMMPS data file with fixed C/B/N type IDs."""
    filename = Path(filename)
    with filename.open("w") as handle:
        handle.write(f"{comment}\n\n")
        handle.write(f"{len(symbols)} atoms\n")
        handle.write("3 atom types\n\n")
        handle.write(f"0.0 {box_lx:.12f} xlo xhi\n")
        handle.write(f"0.0 {box_ly:.12f} ylo yhi\n")
        handle.write(f"0.0 {box_lz:.12f} zlo zhi\n\n")

        handle.write("Masses\n\n")
        handle.write(f"1 {MASS['C']:.8f} # C\n")
        handle.write(f"2 {MASS['B']:.8f} # B\n")
        handle.write(f"3 {MASS['N']:.8f} # N\n")

        handle.write("\nAtoms # atomic\n\n")
        for atom_id, (position, symbol) in enumerate(zip(positions, symbols), start=1):
            handle.write(
                f"{atom_id} {TYPE_ID[symbol]} "
                f"{position[0]:.12f} {position[1]:.12f} {position[2]:.12f}\n"
            )


def write_xyz(filename: str | Path, positions: np.ndarray, symbols: Sequence[str], comment: str) -> None:
    filename = Path(filename)
    with filename.open("w") as handle:
        handle.write(f"{len(symbols)}\n")
        handle.write(f"{comment}\n")
        for symbol, position in zip(symbols, positions):
            handle.write(
                f"{symbol:2s} {position[0]:.10f} "
                f"{position[1]:.10f} {position[2]:.10f}\n"
            )


def write_poscar(
    filename: str | Path,
    positions: np.ndarray,
    symbols: Sequence[str],
    box_lx: float,
    box_ly: float,
    box_lz: float,
    comment: str,
) -> None:
    """Write a VASP POSCAR with atoms grouped in C, B, N order."""
    filename = Path(filename)
    species_order = ["C", "B", "N"]
    ordered_indices = [
        i for species in species_order for i, symbol in enumerate(symbols) if symbol == species
    ]
    counts = [sum(symbol == species for symbol in symbols) for species in species_order]

    cell = np.diag([box_lx, box_ly, box_lz])
    inv_cell = np.linalg.inv(cell)
    fractions = positions[ordered_indices] @ inv_cell
    fractions -= np.floor(fractions)

    with filename.open("w") as handle:
        handle.write(f"{comment}\n")
        handle.write("1.0\n")
        for vector in cell:
            handle.write("  " + "  ".join(f"{x:.12f}" for x in vector) + "\n")
        handle.write("  C  B  N\n")
        handle.write("  " + "  ".join(str(count) for count in counts) + "\n")
        handle.write("Direct\n")
        for frac in fractions:
            handle.write("  " + "  ".join(f"{x:.12f}" for x in frac) + "\n")


def write_group_template(
    filename: str | Path,
    sheet_xlo: float,
    sheet_xhi: float,
    sheet_ylo: float,
    sheet_yhi: float,
    grip_width: float,
) -> None:
    """Write groups/regions that can be included after read_data in LAMMPS."""
    filename = Path(filename)
    y_bottom_hi = sheet_ylo + grip_width
    y_top_lo = sheet_yhi - grip_width

    with filename.open("w") as handle:
        handle.write("# Include this file after read_data\n")
        handle.write("# Atom types: 1=C, 2=B, 3=N\n\n")
        handle.write(f"region lower_grip block {sheet_xlo:.8f} {sheet_xhi:.8f} "
                     f"{sheet_ylo:.8f} {y_bottom_hi:.8f} INF INF units box\n")
        handle.write(f"region upper_grip block {sheet_xlo:.8f} {sheet_xhi:.8f} "
                     f"{y_top_lo:.8f} {sheet_yhi:.8f} INF INF units box\n")
        handle.write("group lower_grip region lower_grip\n")
        handle.write("group upper_grip region upper_grip\n")
        handle.write("group mobile subtract all lower_grip upper_grip\n")
        handle.write("group graphene type 1\n")
        handle.write("group hBN type 2 3\n")


def parse_sizes(values: Iterable[str]) -> List[Tuple[float, float, str]]:
    """Parse W x H values such as 100x200; a scalar creates a square."""
    parsed: List[Tuple[float, float, str]] = []
    for value in values:
        clean = value.lower().replace("angstrom", "").replace("å", "").replace("a", "")
        if "x" in clean:
            first, second = clean.split("x", maxsplit=1)
            sx, sy = float(first), float(second)
        else:
            sx = sy = float(clean)

        if sx <= 0.0 or sy <= 0.0:
            raise ValueError("All requested dimensions must be positive.")

        label = f"{format_number(sx)}x{format_number(sy)}"
        parsed.append((sx, sy, label))
    return parsed


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1.0e-10:
        return str(int(round(value)))
    return str(value).replace(".", "p")


def count_symbols(symbols: Sequence[str]) -> dict[str, int]:
    return {
        element: int(sum(symbol == element for symbol in symbols))
        for element in ("C", "B", "N")
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Generate rectangular graphene sheets with a circular coherent h-BN "
            "inclusion, with or without a horizontal edge crack."
        ),
    )
    parser.add_argument("--graphene_poscar", required=True, help="Graphene POSCAR/CONTCAR")
    parser.add_argument("--bn_poscar", required=True, help="h-BN POSCAR/CONTCAR")
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["100x200"],
        help="Nominal W x H dimensions in Angstrom, e.g. 100x200 200x400",
    )
    parser.add_argument(
        "--variants",
        choices=("both", "no_crack", "with_crack"),
        default="both",
        help="Which geometries to write",
    )

    radius_group = parser.add_mutually_exclusive_group()
    radius_group.add_argument("--radius", type=float, default=None, help="Fixed inclusion radius in A")
    radius_group.add_argument(
        "--radius_fraction",
        type=float,
        default=0.15,
        help="Inclusion radius divided by actual sheet width W",
    )
    parser.add_argument("--center_x", type=float, default=0.50, help="Inclusion center x/W")
    parser.add_argument("--center_y", type=float, default=0.30, help="Inclusion center y/H")
    parser.add_argument("--swap_bn", action="store_true", help="Exchange the B and N sublattices")

    crack_length_group = parser.add_mutually_exclusive_group()
    crack_length_group.add_argument("--crack_length", type=float, default=None, help="Fixed crack length in A")
    crack_length_group.add_argument(
        "--crack_length_fraction",
        type=float,
        default=0.10,
        help="Crack length divided by actual sheet width W",
    )

    crack_y_group = parser.add_mutually_exclusive_group()
    crack_y_group.add_argument(
        "--crack_y",
        type=float,
        default=None,
        help="Crack centerline y coordinate measured from the lower sheet edge, in A",
    )
    crack_y_group.add_argument(
        "--crack_y_fraction",
        type=float,
        default=0.55,
        help="Crack centerline y/H",
    )
    parser.add_argument("--crack_width", type=float, default=2.0, help="Full deleted slit width in A")
    parser.add_argument("--crack_side", choices=("left", "right"), default="left")

    parser.add_argument("--vacuum", type=float, default=25.0, help="Simulation box height in z, in A")
    parser.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help="Empty x/y margin around the sheet in the LAMMPS box, in A",
    )
    parser.add_argument(
        "--grip_width",
        type=float,
        default=5.0,
        help="Top and bottom grip width written to the companion .lmp file, in A",
    )
    parser.add_argument("--outdir", default="Gr_BN_rectangular", help="Output directory")
    return parser


def main() -> None:
    args = argument_parser().parse_args()

    if args.vacuum <= 0.0:
        raise ValueError("--vacuum must be positive.")
    if args.padding < 0.0:
        raise ValueError("--padding cannot be negative.")
    if not (0.0 < args.center_x < 1.0 and 0.0 < args.center_y < 1.0):
        raise ValueError("--center_x and --center_y must lie strictly between 0 and 1.")
    if args.radius is not None and args.radius <= 0.0:
        raise ValueError("--radius must be positive.")
    if args.radius is None and not (0.0 < args.radius_fraction < 0.5):
        raise ValueError("--radius_fraction must be between 0 and 0.5.")
    if args.crack_length is not None and args.crack_length <= 0.0:
        raise ValueError("--crack_length must be positive.")
    if args.crack_length is None and not (0.0 < args.crack_length_fraction < 1.0):
        raise ValueError("--crack_length_fraction must be between 0 and 1.")
    if args.crack_y is None and not (0.0 < args.crack_y_fraction < 1.0):
        raise ValueError("--crack_y_fraction must be between 0 and 1.")
    if args.crack_width <= 0.0:
        raise ValueError("--crack_width must be positive.")
    if args.grip_width <= 0.0:
        raise ValueError("--grip_width must be positive.")

    graphene_raw = read_poscar(args.graphene_poscar)
    bn_raw = read_poscar(args.bn_poscar)

    if set(graphene_raw.symbols) != {"C"}:
        raise ValueError("The graphene POSCAR must contain only C atoms.")
    if set(bn_raw.symbols) != {"B", "N"}:
        raise ValueError("The h-BN POSCAR must contain B and N atoms only.")

    graphene_base = make_rectangular_2d(graphene_raw)
    bn_base = make_rectangular_2d(bn_raw)
    base_colors = graphene_sublattice_labels(graphene_base)

    d_cc = minimum_periodic_distance(graphene_base, {frozenset(("C", "C"))})
    d_bn = minimum_periodic_distance(bn_base, {frozenset(("B", "N"))})
    mismatch = 100.0 * (d_bn - d_cc) / d_cc

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sizes = parse_sizes(args.sizes)
    summary_rows: List[dict[str, object]] = []

    print(f"C-C nearest-neighbour distance : {d_cc:.6f} A")
    print(f"B-N nearest-neighbour distance : {d_bn:.6f} A")
    print(f"Natural h-BN/graphene mismatch : {mismatch:+.3f} %")
    print("Generated h-BN atoms remain coherent with the graphene lattice.\n")

    for target_w, target_h, label in sizes:
        positions, colors, sheet_w, sheet_h, nx, ny = build_sheet(
            graphene_base,
            base_colors,
            target_x=target_w,
            target_y=target_h,
            vacuum=args.vacuum,
        )

        radius = args.radius if args.radius is not None else args.radius_fraction * sheet_w
        crack_length = (
            args.crack_length
            if args.crack_length is not None
            else args.crack_length_fraction * sheet_w
        )
        crack_y = args.crack_y if args.crack_y is not None else args.crack_y_fraction * sheet_h

        if radius >= 0.5 * min(sheet_w, sheet_h):
            raise ValueError(
                f"Inclusion radius {radius:.3f} A is too large for {sheet_w:.3f} x {sheet_h:.3f} A."
            )
        if crack_length >= sheet_w:
            raise ValueError(f"Crack length must be less than the sheet width for {label}.")
        if not (0.0 < crack_y < sheet_h):
            raise ValueError(f"Crack y coordinate lies outside the sheet for {label}.")
        if 2.0 * args.grip_width >= sheet_h:
            raise ValueError(f"Grip regions are too large for the sheet height of {label}.")

        symbols, inclusion_mask, (cx, cy) = assign_bn_inclusion(
            positions,
            colors,
            lx=sheet_w,
            ly=sheet_h,
            radius=radius,
            center_fraction=(args.center_x, args.center_y),
            swap_bn=args.swap_bn,
        )

        # Warn when the nominal inclusion crosses a free sheet edge.
        if cx - radius <= 0.0 or cx + radius >= sheet_w or cy - radius <= 0.0 or cy + radius >= sheet_h:
            print(f"WARNING [{label}]: the circular inclusion intersects a sheet boundary.")

        variants: List[str]
        if args.variants == "both":
            variants = ["no_crack", "edge_crack"]
        elif args.variants == "no_crack":
            variants = ["no_crack"]
        else:
            variants = ["edge_crack"]

        # Add optional empty margin around the finite sheet.
        box_lx = sheet_w + 2.0 * args.padding
        box_ly = sheet_h + 2.0 * args.padding
        shifted_positions = positions.copy()
        shifted_positions[:, 0] += args.padding
        shifted_positions[:, 1] += args.padding

        for variant in variants:
            if variant == "no_crack":
                keep = np.ones(len(positions), dtype=bool)
                removed = np.zeros(len(positions), dtype=bool)
            else:
                keep, removed = edge_crack_keep_mask(
                    positions,
                    lx=sheet_w,
                    crack_length=crack_length,
                    crack_y=crack_y,
                    crack_width=args.crack_width,
                    side=args.crack_side,
                )
                if not np.any(removed):
                    raise RuntimeError(
                        f"No atoms were removed for the crack in {label}. "
                        "Increase --crack_width or adjust --crack_y_fraction."
                    )

            output_positions = shifted_positions[keep]
            output_symbols = symbols[keep].tolist()
            output_inclusion = inclusion_mask[keep]
            counts = count_symbols(output_symbols)

            stem = f"Gr_BN_{label}_{variant}"
            comment = (
                f"Rectangular graphene matrix with circular h-BN inclusion; {variant}; "
                "types 1=C, 2=B, 3=N"
            )
            write_lammps_data(
                outdir / f"{stem}.data",
                output_positions,
                output_symbols,
                box_lx,
                box_ly,
                args.vacuum,
                comment,
            )
            write_xyz(outdir / f"{stem}.xyz", output_positions, output_symbols, comment)
            write_poscar(
                outdir / f"POSCAR_{stem}",
                output_positions,
                output_symbols,
                box_lx,
                box_ly,
                args.vacuum,
                comment,
            )

            crack_removed = int(np.count_nonzero(removed))
            tip_x = crack_length if args.crack_side == "left" else sheet_w - crack_length
            inclusion_tip_distance = math.hypot(tip_x - cx, crack_y - cy)
            nominal_gap = inclusion_tip_distance - radius

            summary_rows.append(
                {
                    "requested_W_A": target_w,
                    "requested_H_A": target_h,
                    "actual_W_A": sheet_w,
                    "actual_H_A": sheet_h,
                    "box_Lx_A": box_lx,
                    "box_Ly_A": box_ly,
                    "box_Lz_A": args.vacuum,
                    "repeat_nx": nx,
                    "repeat_ny": ny,
                    "variant": variant,
                    "total_atoms": len(output_symbols),
                    "C_atoms": counts["C"],
                    "B_atoms": counts["B"],
                    "N_atoms": counts["N"],
                    "BN_inclusion_atoms_remaining": int(np.count_nonzero(output_inclusion)),
                    "inclusion_radius_A": radius,
                    "inclusion_center_x_A": cx,
                    "inclusion_center_y_A": cy,
                    "crack_side": args.crack_side if variant == "edge_crack" else "none",
                    "crack_length_A": crack_length if variant == "edge_crack" else 0.0,
                    "crack_y_A": crack_y if variant == "edge_crack" else 0.0,
                    "crack_width_A": args.crack_width if variant == "edge_crack" else 0.0,
                    "crack_removed_atoms": crack_removed,
                    "crack_tip_to_inclusion_center_A": inclusion_tip_distance,
                    "nominal_tip_to_inclusion_boundary_gap_A": nominal_gap,
                    "CC_bond_A": d_cc,
                    "BN_natural_bond_A": d_bn,
                    "BN_graphene_mismatch_percent": mismatch,
                    "data_file": f"{stem}.data",
                }
            )

            print(
                f"{stem}: sheet={sheet_w:.3f} x {sheet_h:.3f} A, "
                f"atoms={len(output_symbols)}, C/B/N={counts['C']}/{counts['B']}/{counts['N']}, "
                f"crack-removed={crack_removed}"
            )

        write_group_template(
            outdir / f"Gr_BN_{label}_groups.lmp",
            sheet_xlo=args.padding,
            sheet_xhi=args.padding + sheet_w,
            sheet_ylo=args.padding,
            sheet_yhi=args.padding + sheet_h,
            grip_width=args.grip_width,
        )

    summary_file = outdir / "generation_summary.csv"
    with summary_file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nFiles written to: {outdir.resolve()}")
    print("LAMMPS type mapping: 1=C, 2=B, 3=N")
    print("Use pair_coeff element order: C B N")


if __name__ == "__main__":
    main()
