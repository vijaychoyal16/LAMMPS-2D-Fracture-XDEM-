#!/usr/bin/env python3
"""
Create X-shaped / cross-shaped central cracks in 2D materials from POSCAR.

Works for:
    graphene
    BN
    BC3
    C3N
    BCN
    C2N
    Bi2Se3
    any 2D POSCAR with vacuum along z

Output for each size:
    POSCAR_100x100_cross_crack
    data_100x100_cross_crack.data
    preview_100x100_cross_crack.png
    summary_cross_cracks.txt
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from ase.io import read, write
from ase.build import make_supercell


# ============================================================
# Geometry helper functions
# ============================================================

def angle_between(v1, v2):
    """Return angle in degrees between two vectors."""
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)

    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cosang = np.clip(cosang, -1.0, 1.0)

    return np.degrees(np.arccos(cosang))


def is_hexagonal_2d(atoms, tol_angle=3.0, tol_length=0.05):
    """
    Detect graphene/h-BN-like primitive hexagonal 2D cell.
    """
    cell = atoms.cell.array
    a = cell[0, :2]
    b = cell[1, :2]

    la = np.linalg.norm(a)
    lb = np.linalg.norm(b)
    gamma = angle_between(a, b)

    length_close = abs(la - lb) / max(la, lb) < tol_length
    angle_hex = abs(gamma - 60.0) < tol_angle or abs(gamma - 120.0) < tol_angle

    return length_close and angle_hex


def convert_hex_to_rectangular(atoms):
    """
    Convert hexagonal primitive 2D cell to rectangular-like cell.

    For graphene/BN primitive cell:
        new_a = old_a + old_b
        new_b = old_a - old_b
    """
    P = np.array([
        [1,  1, 0],
        [1, -1, 0],
        [0,  0, 1]
    ])

    return make_supercell(atoms, P)


def rotate_cell_to_xy(atoms):
    """
    Rotate atoms/cell so the first lattice vector is along x direction.
    """
    atoms = atoms.copy()
    cell = atoms.cell.array

    a = cell[0, :2]
    theta = np.arctan2(a[1], a[0])

    c = np.cos(-theta)
    s = np.sin(-theta)

    R = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])

    atoms.set_positions(atoms.get_positions() @ R.T)
    atoms.set_cell(cell @ R.T, scale_atoms=False)
    atoms.wrap()

    return atoms


def center_2d_sheet(atoms, vacuum_z=25.0):
    """
    Reset z cell length and center the 2D slab along z.
    """
    atoms = atoms.copy()
    cell = atoms.cell.array.copy()

    z = atoms.positions[:, 2]
    zmin = z.min()
    zmax = z.max()
    slab_thickness = zmax - zmin

    new_lz = slab_thickness + vacuum_z
    cell[2] = [0.0, 0.0, new_lz]

    atoms.set_cell(cell, scale_atoms=False)

    old_center = 0.5 * (zmin + zmax)
    new_center = 0.5 * new_lz

    atoms.positions[:, 2] += new_center - old_center
    atoms.wrap()

    return atoms


def make_square_supercell(atoms, target_size, exact_scale=False):
    """
    Make a square-like 2D supercell.

    If exact_scale=False:
        repeat until Lx and Ly are at least target_size.
        Original lattice constants are preserved.

    If exact_scale=True:
        scale final Lx and Ly exactly to target_size.
        This introduces small artificial strain.
    """
    atoms = atoms.copy()

    cell = atoms.cell.array
    lx = np.linalg.norm(cell[0, :2])
    ly = np.linalg.norm(cell[1, :2])

    nx = int(np.ceil(target_size / lx))
    ny = int(np.ceil(target_size / ly))

    atoms = atoms.repeat((nx, ny, 1))

    if exact_scale:
        cell = atoms.cell.array.copy()

        current_lx = np.linalg.norm(cell[0, :2])
        current_ly = np.linalg.norm(cell[1, :2])

        sx = target_size / current_lx
        sy = target_size / current_ly

        pos = atoms.get_positions()
        pos[:, 0] *= sx
        pos[:, 1] *= sy

        cell[0, :] *= sx
        cell[1, :] *= sy

        atoms.set_cell(cell, scale_atoms=False)
        atoms.set_positions(pos)

    atoms.wrap()

    return atoms, nx, ny


# ============================================================
# Crack generation functions
# ============================================================

def distance_to_segment(points, p1, p2):
    """
    Calculate distance from many 2D points to one finite line segment p1-p2.
    """
    points = np.asarray(points, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    v = p2 - p1
    vv = np.dot(v, v)

    if vv < 1.0e-12:
        raise ValueError("Crack segment length is too small.")

    t = np.dot(points - p1, v) / vv
    t_clip = np.clip(t, 0.0, 1.0)

    closest = p1 + t_clip[:, None] * v
    dist = np.linalg.norm(points - closest, axis=1)

    return dist


def create_cross_crack(
    atoms,
    crack_length,
    crack_width,
    angle_deg,
    center_xy=None,
    second_angle_deg=None
):
    """
    Create X-shaped crack by deleting atoms close to two crossing line segments.

    crack_length:
        total length of each crack line, equivalent to 2a.

    crack_width:
        deletion width around each crack line.

    angle_deg:
        angle beta of first crack with respect to +x axis.

    second_angle_deg:
        angle of second crack.
        If None, use -angle_deg, giving symmetric X crack.

    center_xy:
        crack center in Angstrom.
        If None, use center of simulation cell.
    """

    atoms = atoms.copy()
    cell = atoms.cell.array

    a_vec = cell[0, :2]
    b_vec = cell[1, :2]

    if center_xy is None:
        center_xy = 0.5 * (a_vec + b_vec)
    else:
        center_xy = np.array(center_xy, dtype=float)

    if second_angle_deg is None:
        second_angle_deg = -angle_deg

    beta1 = np.deg2rad(angle_deg)
    beta2 = np.deg2rad(second_angle_deg)

    direction1 = np.array([np.cos(beta1), np.sin(beta1)])
    direction2 = np.array([np.cos(beta2), np.sin(beta2)])

    p1a = center_xy - 0.5 * crack_length * direction1
    p1b = center_xy + 0.5 * crack_length * direction1

    p2a = center_xy - 0.5 * crack_length * direction2
    p2b = center_xy + 0.5 * crack_length * direction2

    xy = atoms.positions[:, :2]

    dist1 = distance_to_segment(xy, p1a, p1b)
    dist2 = distance_to_segment(xy, p2a, p2b)

    remove_mask = (dist1 <= 0.5 * crack_width) | (dist2 <= 0.5 * crack_width)

    removed_indices = np.where(remove_mask)[0]

    cracked_atoms = atoms.copy()
    del cracked_atoms[removed_indices]
    cracked_atoms.wrap()

    crack_info = {
        "center": center_xy,
        "angle1_deg": angle_deg,
        "angle2_deg": second_angle_deg,
        "length": crack_length,
        "width": crack_width,
        "p1a": p1a,
        "p1b": p1b,
        "p2a": p2a,
        "p2b": p2b,
        "initial_atoms": len(atoms),
        "final_atoms": len(cracked_atoms),
        "removed_atoms": len(removed_indices)
    }

    return cracked_atoms, crack_info


def plot_cross_crack_preview(atoms_before, atoms_after, crack_info, filename):
    """
    Save x-y preview image of the cross crack.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    xy_before = atoms_before.positions[:, :2]
    xy_after = atoms_after.positions[:, :2]

    ax.scatter(
        xy_before[:, 0],
        xy_before[:, 1],
        s=1,
        alpha=0.15,
        label="Original atoms"
    )

    ax.scatter(
        xy_after[:, 0],
        xy_after[:, 1],
        s=1,
        alpha=0.85,
        label="Remaining atoms"
    )

    p1a = crack_info["p1a"]
    p1b = crack_info["p1b"]
    p2a = crack_info["p2a"]
    p2b = crack_info["p2b"]

    ax.plot([p1a[0], p1b[0]], [p1a[1], p1b[1]], linewidth=2.5, label="Crack 1")
    ax.plot([p2a[0], p2b[0]], [p2a[1], p2b[1]], linewidth=2.5, label="Crack 2")

    ax.set_aspect("equal")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_title(
        f"Cross crack: length={crack_info['length']:.2f} Å, "
        f"width={crack_info['width']:.2f} Å, "
        f"angles={crack_info['angle1_deg']:.1f}°, {crack_info['angle2_deg']:.1f}°"
    )

    ax.legend(markerscale=6, frameon=False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def write_lammps_data(atoms, filename, specorder=None):
    """
    Write LAMMPS data file.

    specorder example:
        graphene: --specorder C
        BN:       --specorder B N
        BC3:      --specorder B C
        C3N:      --specorder C N
        BCN:      --specorder B C N
    """
    symbols = atoms.get_chemical_symbols()

    if specorder is not None:
        order = []
        for el in specorder:
            if el in symbols:
                order.append(el)

        for el in sorted(set(symbols)):
            if el not in order:
                order.append(el)
    else:
        order = sorted(set(symbols))

    write(
        filename,
        atoms,
        format="lammps-data",
        atom_style="atomic",
        masses=True,
        specorder=order
    )

    return order


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate X-shaped cross cracks from POSCAR for 2D materials."
    )

    parser.add_argument(
        "-i", "--input",
        default="POSCAR",
        help="Input POSCAR file."
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        type=float,
        default=[100, 200, 300, 400, 500],
        help="Target sheet sizes in Angstrom."
    )

    parser.add_argument(
        "--vacuum",
        type=float,
        default=25.0,
        help="Vacuum thickness along z in Angstrom."
    )

    parser.add_argument(
        "--angle",
        type=float,
        default=75.0,
        help="First crack angle beta in degrees."
    )

    parser.add_argument(
        "--second-angle",
        type=float,
        default=None,
        help="Second crack angle in degrees. Default: -angle."
    )

    parser.add_argument(
        "--crack-length",
        type=float,
        default=None,
        help="Absolute crack length in Angstrom. If not given, use crack-fraction."
    )

    parser.add_argument(
        "--crack-fraction",
        type=float,
        default=0.35,
        help="Crack length as fraction of min(Lx,Ly). Example: 0.35"
    )

    parser.add_argument(
        "--crack-width",
        type=float,
        default=3.0,
        help="Deletion width around each crack in Angstrom."
    )

    parser.add_argument(
        "--center",
        nargs=2,
        type=float,
        default=None,
        metavar=("X", "Y"),
        help="Crack center in Angstrom. Default: cell center."
    )

    parser.add_argument(
        "--exact",
        action="store_true",
        help="Scale final Lx and Ly exactly to target size."
    )

    parser.add_argument(
        "--no-hex-convert",
        action="store_true",
        help="Do not convert hexagonal primitive cell to rectangular cell."
    )

    parser.add_argument(
        "--write-perfect",
        action="store_true",
        help="Also write perfect uncracked supercells."
    )

    parser.add_argument(
        "--specorder",
        nargs="+",
        default=None,
        help="LAMMPS element order, e.g. --specorder B C N"
    )

    parser.add_argument(
        "--outdir",
        default="cross_crack_supercells",
        help="Output directory."
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    atoms = read(args.input, format="vasp")

    print("\nInput structure")
    print("----------------")
    print(f"File: {args.input}")
    print(f"Atoms: {len(atoms)}")
    print(f"Elements: {sorted(set(atoms.get_chemical_symbols()))}")
    print("Cell:")
    print(atoms.cell)

    if not args.no_hex_convert and is_hexagonal_2d(atoms):
        print("\nDetected hexagonal 2D primitive cell.")
        print("Converting to rectangular cell...")
        atoms = convert_hex_to_rectangular(atoms)
    else:
        print("\nUsing original in-plane cell shape.")

    atoms = rotate_cell_to_xy(atoms)
    atoms = center_2d_sheet(atoms, vacuum_z=args.vacuum)

    summary_file = os.path.join(args.outdir, "summary_cross_cracks.txt")

    with open(summary_file, "w") as f:
        f.write(
            "target_A nx ny atoms_before atoms_after removed_atoms "
            "Lx_A Ly_A Lz_A crack_length_A crack_width_A "
            "angle1_deg angle2_deg "
            "crack1_x1 crack1_y1 crack1_x2 crack1_y2 "
            "crack2_x1 crack2_y1 crack2_x2 crack2_y2 "
            "lammps_order\n"
        )

        for target in args.sizes:
            supercell, nx, ny = make_square_supercell(
                atoms,
                target_size=target,
                exact_scale=args.exact
            )

            cell = supercell.cell.array
            lx = np.linalg.norm(cell[0, :2])
            ly = np.linalg.norm(cell[1, :2])
            lz = np.linalg.norm(cell[2])

            if args.crack_length is None:
                crack_length = args.crack_fraction * min(lx, ly)
            else:
                crack_length = args.crack_length

            cracked, info = create_cross_crack(
                supercell,
                crack_length=crack_length,
                crack_width=args.crack_width,
                angle_deg=args.angle,
                second_angle_deg=args.second_angle,
                center_xy=args.center
            )

            tag = (
                f"{int(target)}x{int(target)}_cross_"
                f"b{args.angle:g}_L{crack_length:.1f}_W{args.crack_width:.1f}"
            )

            poscar_crack = os.path.join(args.outdir, f"POSCAR_{tag}_crack")
            data_crack = os.path.join(args.outdir, f"data_{tag}_crack.data")
            png_crack = os.path.join(args.outdir, f"preview_{tag}_crack.png")

            write(
                poscar_crack,
                cracked,
                format="vasp",
                direct=True,
                sort=True,
                vasp5=True
            )

            order = write_lammps_data(
                cracked,
                data_crack,
                specorder=args.specorder
            )

            plot_cross_crack_preview(
                supercell,
                cracked,
                info,
                png_crack
            )

            if args.write_perfect:
                poscar_perfect = os.path.join(args.outdir, f"POSCAR_{int(target)}x{int(target)}_perfect")
                data_perfect = os.path.join(args.outdir, f"data_{int(target)}x{int(target)}_perfect.data")

                write(
                    poscar_perfect,
                    supercell,
                    format="vasp",
                    direct=True,
                    sort=True,
                    vasp5=True
                )

                write_lammps_data(
                    supercell,
                    data_perfect,
                    specorder=args.specorder
                )

            f.write(
                f"{target:.2f} {nx:d} {ny:d} "
                f"{info['initial_atoms']:d} {info['final_atoms']:d} {info['removed_atoms']:d} "
                f"{lx:.6f} {ly:.6f} {lz:.6f} "
                f"{info['length']:.6f} {info['width']:.6f} "
                f"{info['angle1_deg']:.6f} {info['angle2_deg']:.6f} "
                f"{info['p1a'][0]:.6f} {info['p1a'][1]:.6f} "
                f"{info['p1b'][0]:.6f} {info['p1b'][1]:.6f} "
                f"{info['p2a'][0]:.6f} {info['p2a'][1]:.6f} "
                f"{info['p2b'][0]:.6f} {info['p2b'][1]:.6f} "
                f"{','.join(order)}\n"
            )

            print("\nGenerated cross-crack structure")
            print("--------------------------------")
            print(f"Target size       : {target:.1f} x {target:.1f} Å")
            print(f"Actual Lx, Ly, Lz : {lx:.3f}, {ly:.3f}, {lz:.3f} Å")
            print(f"Repeat            : {nx} x {ny} x 1")
            print(f"Atoms before      : {info['initial_atoms']}")
            print(f"Atoms after       : {info['final_atoms']}")
            print(f"Removed atoms     : {info['removed_atoms']}")
            print(f"Crack length      : {info['length']:.3f} Å")
            print(f"Crack width       : {info['width']:.3f} Å")
            print(f"Angle 1           : {info['angle1_deg']:.3f} degree")
            print(f"Angle 2           : {info['angle2_deg']:.3f} degree")
            print(f"LAMMPS order      : {order}")
            print(f"POSCAR            : {poscar_crack}")
            print(f"LAMMPS data       : {data_crack}")
            print(f"Preview image     : {png_crack}")

    print("\nDone.")
    print(f"Summary file: {summary_file}")


if __name__ == "__main__":
    main()
