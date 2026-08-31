# LAMMPS-2D-Fracture-XDEM-

# Atomistic Fracture Mechanics of 2D Materials Using LAMMPS and Machine-Learning Interatomic Potentials

## Overview

This repository contains a computational framework for studying the atomistic fracture behavior of graphene and other two-dimensional (2D) materials using **LAMMPS** coupled with **machine-learning interatomic potentials (MLIPs/MTPs)**.

The framework was developed to investigate how crack geometry, crack orientation, loading mode, boundary conditions, and mixed-mode deformation influence crack initiation and propagation in 2D materials.

The implemented configurations include:

- Pristine tensile deformation
- Horizontal edge cracks
- Inclined cracks
- Cross/X-shaped cracks
- Mode-I tensile fracture
- Mode-II shear fracture
- 70° kinked/inclined crack propagation
- Mixed Mode-I + Mode-II fracture
- Finite-sheet nonperiodic boundary conditions
- 0 K quasi-static fracture simulations
- Stress-strain and reaction-force extraction
- Atomistic crack visualization using OVITO

The simulations are designed primarily for graphene but can be extended to other 2D materials by replacing the atomic structure, effective thickness, and interatomic potential.

---

## 1. Computational Methodology

The fracture calculations are performed using a **0 K quasi-static displacement-controlled approach**.

At every loading increment:

1. A small displacement is imposed on the prescribed boundary atoms.
2. The remaining unconstrained atoms are allowed to relax.
3. Energy minimization is performed.
4. Reaction forces and virial stresses are calculated.
5. Atomic configurations are stored.
6. The loading increment is repeated until crack propagation or complete fracture occurs.

The general workflow is

```text
Atomic structure
      |
      v
Crack generation
      |
      v
LAMMPS data file
      |
      v
MLIP/MTP interatomic potential
      |
      v
Initial 0 K minimization
      |
      v
Incremental displacement
      |
      v
Atomic relaxation
      |
      v
Reaction force + virial stress
      |
      v
Stress-strain response
      |
      v
Crack initiation and propagation
      |
      v
OVITO visualization
```

This quasi-static approach avoids introducing thermal effects and enables direct investigation of crack-tip deformation and fracture processes.

---

# 2. Repository Structure

```text
Atomistic-Fracture-2D-Materials-LAMMPS/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── inputs/
│   │
│   ├── pristine_tension/
│   │   └── tension_y_0K_periodic.in
│   │
│   ├── horizontal_edge_crack/
│   │   ├── crack_y_0K_top_pull.in
│   │   └── crack_0K_finite_top_pull_ffp.in
│   │
│   ├── inclined_crack/
│   │   └── inclined_crack_y_top_pull_0K.in
│   │
│   ├── cross_crack/
│   │   └── cross_crack_y_top_pull.in
│   │
│   ├── modeII_edge_crack/
│   │   ├── crack_x_0K_top_pull_modeII.in
│   │   └── crack_x_0K_top_pull_modeII_affine.in
│   │
│   ├── kinked_70deg_crack/
│   │   └── crack_x_0K_toponly_theta70_fracture.in
│   │
│   └── mixed_mode/
│       └── crack_0K_mixed_tension_shear_theta70.in
│
├── structures/
│   ├── README.md
│   └── examples/
│
├── potentials/
│   └── README.md
│
├── scripts/
│   ├── structure_generation/
│   ├── postprocessing/
│   └── plotting/
│
├── examples/
│   └── graphene_100x100/
│
└── docs/
    ├── boundary_conditions.md
    ├── crack_geometries.md
    └── workflow.md
```

---

# 3. Interatomic Potential

The calculations use a machine-learning interatomic potential through the LAMMPS MLIP interface.

A typical potential definition is

```lammps
pair_style      mlip mlip.ini
pair_coeff      * * mlip
```

The trained MTP/MLIP must provide an accurate description not only of equilibrium graphene but also of highly strained and fracture-related atomic configurations.

The training database should therefore ideally contain configurations involving:

- equilibrium structures,
- elastic deformation,
- tensile deformation,
- shear deformation,
- highly strained C-C bonds,
- distorted local environments,
- crack-tip-like environments,
- surfaces and edges,
- bond-breaking configurations.

Fracture predictions should not be considered reliable if the potential was trained only around equilibrium configurations.

---

# 4. Effective Thickness of Graphene

For converting atomistic virial quantities into an effective three-dimensional stress, an effective graphene thickness of

```text
h = 3.35 Å
```

is used.

For example,

```lammps
variable h equal 3.35
```

This convention should be clearly reported whenever stresses are presented in GPa.

For strictly 2D mechanical properties, reaction forces may alternatively be reported in units such as N/m.

---

# 5. Pristine Tensile Deformation

### Input

```text
inputs/pristine_tension/tension_y_0K_periodic.in
```

This calculation provides the reference tensile response of the pristine material.

The Y direction is incrementally stretched and the atomic structure is minimized after each deformation increment.

The simulation can be used to obtain:

- Young's modulus
- tensile stress-strain curve
- ultimate tensile strength
- failure strain
- potential-energy evolution
- reference mechanical response for comparison with cracked structures

A periodic implementation may use

```lammps
boundary p p p
```

together with incremental box deformation.

This calculation is mainly intended as a **reference bulk/pristine calculation** rather than the preferred boundary condition for finite cracked specimens.

---

# 6. Horizontal Edge Crack — Mode-I Fracture

### Input

```text
inputs/horizontal_edge_crack/crack_y_0K_top_pull.in
```

A horizontal crack originates from the left edge of the graphene sheet.

A schematic representation is

```text
                 +uy
          ↑ ↑ ↑ ↑ ↑ ↑ ↑
       ┌─────────────────────┐
       │                     │
       │                     │
       │                     │
       │──────────●          │
       │   crack    tip      │
       │                     │
       │                     │
       └─────────────────────┘
             fixed bottom
```

The loading primarily produces **Mode-I crack opening**.

The bottom atomic strip is constrained while the upper atomic strip is incrementally displaced in the positive Y direction.

A typical loading operation is

```lammps
displace_atoms top move 0.0 ${dy_step} 0.0 units box
```

After every displacement increment, energy minimization is performed.

---

# 7. Finite Horizontal Edge-Crack Model

### Recommended input

```text
inputs/horizontal_edge_crack/crack_0K_finite_top_pull_ffp.in
```

For finite graphene fracture specimens, the preferred boundary condition is

```lammps
boundary f f p
```

corresponding to

```text
X direction : nonperiodic
Y direction : nonperiodic
Z direction : periodic
```

The Z direction contains vacuum and is not mechanically loaded.

This configuration prevents artificial periodic connection between the top and bottom edges of the graphene sheet.

The boundary conditions are

```text
Top grip       → prescribed displacement
Bottom grip    → fixed
Left edge      → free
Right edge     → free
```

This implementation is preferred for finite-sheet edge-crack fracture simulations.

---

# 8. Inclined Crack Simulations

### Input

```text
inputs/inclined_crack/inclined_crack_y_top_pull_0K.in
```

The framework can investigate orientation-dependent fracture using cracks with different initial angles.

Example orientations include

```text
15°
30°
45°
60°
75°
```

A representative configuration is

```text
          ↑ ↑ ↑ ↑ ↑ ↑ ↑
       ┌───────────────────┐
       │                   │
       │        /          │
       │       /           │
       │      / crack      │
       │                   │
       │                   │
       └───────────────────┘
             fixed
```

The crack angle is introduced during structure generation rather than by modifying the interatomic potential.

These simulations can be used to investigate:

- orientation-dependent fracture strength,
- crack-tip stress concentration,
- crack deflection,
- anisotropic crack propagation,
- lattice-dependent fracture trajectories.

---

# 9. Cross/X-Shaped Crack

### Input

```text
inputs/cross_crack/cross_crack_y_top_pull.in
```

The cross-crack configuration contains two intersecting cracks.

A schematic configuration is

```text
          ↑ ↑ ↑ ↑ ↑ ↑
       ┌──────────────────┐
       │                  │
       │      \  /        │
       │       \/         │
       │       /\         │
       │      /  \        │
       │                  │
       └──────────────────┘
              fixed
```

This geometry is useful for studying:

- crack-crack interaction,
- competing crack tips,
- stress redistribution,
- crack coalescence,
- dominant crack selection,
- fracture-path competition.

The geometry should be generated before the LAMMPS calculation and supplied as a LAMMPS data file.

---

# 10. Mode-II Edge-Crack Shear

### Input

```text
inputs/modeII_edge_crack/crack_x_0K_top_pull_modeII.in
```

Mode-II loading is generated by displacing the upper grip horizontally.

```text
          → → → → → → →
       ┌─────────────────────┐
       │                     │
       │                     │
       │──────────●          │
       │   crack             │
       │                     │
       └─────────────────────┘
             fixed
```

The loading direction is

```text
Top edge:
ux > 0

Bottom edge:
ux = 0
uy = 0
```

A typical loading command is

```lammps
displace_atoms top move ${dx_step} 0.0 0.0 units box
```

The top-edge reaction force is calculated using

```lammps
compute Ftop top reduce sum fx fy fz
```

The X component provides the reaction shear force.

This simulation can be used to study:

- Mode-II fracture,
- crack-tip sliding,
- shear-induced crack initiation,
- inclined crack growth,
- competition between sliding and bond breaking.

---

# 11. Affine Mode-II Shear

### Input

```text
inputs/modeII_edge_crack/crack_x_0K_top_pull_modeII_affine.in
```

An alternative Mode-II implementation distributes the horizontal displacement continuously through the specimen height.

For example,

```lammps
variable lambda atom ...
variable dx_affine atom ...
displace_atoms deform move v_dx_affine 0.0 0.0 units box
```

The displacement approximately follows

```text
ux(y) ∝ y
```

instead of moving only the upper grip.

This produces a smoother shear field.

However, affine deformation distributes deformation over the entire sheet and can reduce crack-tip localization compared with top-grip-only loading.

Therefore, both implementations are retained for comparison.

---

# 12. 70° Kinked Crack Under Shear

### Input

```text
inputs/kinked_70deg_crack/crack_x_0K_toponly_theta70_fracture.in
```

This model investigates the development of an inclined crack from an initial edge crack under predominantly shear loading.

The target geometry is approximately

```text
────────────────●
                 \
                  \
                   \ 70°
                    \
                     \
```

The upper grip is displaced horizontally while the bottom grip remains fixed.

Top-only loading produces stronger crack-tip localization than a fully affine displacement field.

A predefined kink/notch can also be introduced into the initial atomic structure.

Importantly, an exact 70° atomistic crack path is **not guaranteed solely by the macroscopic loading direction**.

The actual fracture trajectory depends on:

- graphene lattice orientation,
- crack-tip atomic structure,
- loading-mode ratio,
- notch geometry,
- local bond-breaking energetics,
- interatomic potential accuracy.

Therefore, the 70° configuration should be interpreted as a targeted/preferred crack orientation rather than an automatically enforced fracture direction.

---

# 13. Mixed Mode-I + Mode-II Fracture

### Input

```text
inputs/mixed_mode/crack_0K_mixed_tension_shear_theta70.in
```

A mixed-mode loading implementation was developed to simultaneously apply tensile opening and shear deformation.

The upper boundary is displaced in both X and Y:

```lammps
displace_atoms top move ${dx_step} ${dy_step} 0.0 units box
```

Therefore,

```text
Δux → Mode-II contribution
Δuy → Mode-I contribution
```

The relative magnitude is controlled by

```lammps
variable modeI_ratio equal 0.25
```

where

```text
modeI_ratio = Δuy / Δux
```

Example values are

```text
modeI_ratio = 0.00  → pure Mode-II
modeI_ratio = 0.10  → strongly shear dominated
modeI_ratio = 0.25  → shear-dominated mixed mode
modeI_ratio = 0.50  → stronger Mode-I contribution
modeI_ratio = 1.00  → equal X/Y displacement increments
```

The geometry can be represented as

```text
                ↗ ↗ ↗ ↗ ↗
           shear + tension

       ┌──────────────────────┐
       │                      │
       │                      │
       │─────────●            │
       │          \           │
       │           \ ~70°     │
       │            \         │
       │                      │
       └──────────────────────┘
              fixed
```

This implementation is intended to investigate crack deflection and kinked crack propagation under combined loading.

---

# 14. Quasi-Static Loading

The loading is performed incrementally.

For example,

```lammps
variable dstrain equal 0.001
```

or for shear

```lammps
variable dgamma equal 0.00025
```

After each increment:

```lammps
min_style cg
minimize 1.0e-10 1.0e-10 30000 300000
```

is performed.

A smaller displacement increment generally provides better resolution of:

- crack initiation,
- peak stress,
- unstable crack growth,
- stress drops,
- bond-breaking events.

However, smaller increments substantially increase computational cost.

---

# 15. Reaction-Force Stress

For finite-sheet calculations, boundary reaction forces are particularly useful.

The total force acting on the top grip is calculated using

```lammps
compute Ftop top reduce sum fx fy fz
```

For Mode-I loading,

```text
Fy → tensile reaction force
```

while for Mode-II loading,

```text
Fx → shear reaction force
```

Using an effective width `W` and thickness `h`, an effective stress can be calculated as

```text
σyy = Fy / (W h)
```

and

```text
τxy = Fx / (W h)
```

For graphene,

```text
h = 3.35 Å
```

is used in the present implementation.

---

# 16. Virial Stress

Atomic virial stresses can additionally be calculated using

```lammps
compute Satom all stress/atom NULL virial
```

and the global virial pressure tensor using

```lammps
compute Pvir all pressure NULL virial
```

The effective graphene stress is corrected for the vacuum-containing simulation-cell thickness.

This allows comparison between:

- reaction-force stress,
- virial stress,
- local crack-tip atomic stress.

For finite cracked structures, the boundary reaction force is particularly useful for constructing global load-displacement or effective stress-strain curves.

---

# 17. Output Quantities

Depending on the loading configuration, output files contain quantities such as

```text
strain_y
gamma_xy
top_displacement
top_ux
top_uy

reaction_syy_GPa_corrected
reaction_tauxy_GPa_corrected
reaction_resultant_GPa_corrected

virial_sxx_GPa
virial_syy_GPa
virial_sxy_GPa

Ftop_x
Ftop_y

PE_total_eV
PE_per_atom_eV
dPE_per_atom_eV

Lx
Ly
Lz
natoms
```

These quantities can be used to determine:

- elastic modulus,
- shear response,
- fracture strength,
- critical strain,
- peak load,
- energy evolution,
- onset of crack propagation,
- post-fracture softening.

---

# 18. OVITO Visualization

LAMMPS trajectories are written in a form compatible with OVITO.

Example:

```lammps
write_dump all custom dump_fracture.lammpstrj \
           id type x y z c_PEatom \
           c_Satom[1] c_Satom[2] c_Satom[3] \
           c_Satom[4] c_Satom[5] c_Satom[6]
```

The resulting trajectory can be used to visualize:

- crack opening,
- crack-tip propagation,
- atomic rearrangement,
- local potential energy,
- local stress concentration,
- bond-breaking events.

For finite specimens using

```lammps
boundary f f p
```

periodic images in the X and Y directions should not be displayed.

---

# 19. Crack Geometry Generation

The crack geometry is created before the LAMMPS fracture calculation.

The workflow is

```text
Pristine graphene
       |
       v
Generate required supercell
       |
       v
Define crack center/tip
       |
       v
Define crack orientation
       |
       v
Remove atoms inside crack/notch region
       |
       v
LAMMPS data file
```

Supported geometries include:

```text
Horizontal edge crack

────────────●


Inclined crack

       /


Cross crack

       \ /
        X
       / \


Kinked crack

────────●
         \
          \
```

The crack width should be chosen carefully.

An excessively large notch removes too many atoms and changes the specimen geometry, while an excessively narrow notch may reconstruct during minimization or fail to behave as a physically meaningful crack.

---

# 20. Boundary Conditions

Different boundary conditions are used depending on the simulation.

### Periodic pristine calculations

```lammps
boundary p p p
```

can be used for bulk-like tensile calculations.

### Finite cracked sheets

The preferred configuration is

```lammps
boundary f f p
```

where X and Y are finite/nonperiodic and Z remains periodic with sufficient vacuum.

This prevents artificial interactions between opposite in-plane boundaries.

---

# 21. Running the Simulations

A typical parallel LAMMPS calculation can be executed using

```bash
mpirun -np 32 /path/to/lammps/src/lmp_mpi \
    -in inputs/horizontal_edge_crack/crack_0K_finite_top_pull_ffp.in
```

For the mixed-mode calculation:

```bash
mpirun -np 32 /path/to/lammps/src/lmp_mpi \
    -in inputs/mixed_mode/crack_0K_mixed_tension_shear_theta70.in
```

The exact number of MPI processes should be selected according to system size and available computational resources.

---

# 22. Example Simulation Workflow

For a 100 × 100 Å graphene sheet containing a horizontal edge crack:

```text
Step 1
Generate pristine graphene supercell

Step 2
Introduce horizontal/inclined/cross crack

Step 3
Convert the structure to LAMMPS data format

Step 4
Check the structure in OVITO

Step 5
Load MLIP/MTP potential

Step 6
Perform initial 0 K minimization

Step 7
Define top and bottom grip atoms

Step 8
Incrementally apply displacement

Step 9
Minimize atomic energy after each increment

Step 10
Calculate reaction force and stress

Step 11
Save atomic configurations

Step 12
Identify crack initiation and failure strain

Step 13
Visualize crack propagation in OVITO

Step 14
Post-process stress-strain and energy curves
```

---

# 23. Implemented Loading Cases

| Configuration | Loading | Main purpose |
|---|---|---|
| Pristine graphene | Y tension | Reference mechanical properties |
| Horizontal edge crack | Mode-I | Opening fracture |
| Inclined crack | Y tension | Orientation-dependent fracture |
| Cross crack | Y tension | Crack interaction |
| Horizontal edge crack | Mode-II | Shear fracture |
| Edge crack + affine shear | Mode-II | Distributed shear deformation |
| 70° kinked crack | Mode-II dominant | Inclined crack propagation |
| 70° kinked crack | Mode-I + Mode-II | Mixed-mode fracture |
| Finite cracked sheet | Top pull | Nonperiodic fracture model |

---

# 24. Important Physical Considerations

## Crack direction

The prescribed initial crack angle does not necessarily determine the final propagation angle.

At the atomistic scale, crack growth is influenced by the crystallographic structure and local bond-breaking energetics.

## MLIP transferability

Fracture simulations probe configurations far from equilibrium. Therefore, MLIP accuracy should be validated for highly strained and bond-breaking environments.

## Loading increment

Large deformation increments may skip important fracture events. Smaller increments provide better resolution but require more minimization steps.

## Finite-size effects

Mechanical properties and crack trajectories should be checked for different sheet dimensions.

Example sizes include

```text
100 × 100 Å
100 × 200 Å
200 × 200 Å
300 × 300 Å
500 × 500 Å
```

## Boundary effects

Cracks should remain sufficiently far from artificial grip regions unless an edge-crack configuration specifically requires otherwise.

---

# 25. Current Research Directions

The framework is being extended toward:

- size-dependent fracture of 2D materials,
- orientation-dependent crack propagation,
- mixed-mode fracture,
- crack-defect interaction,
- crack-inclusion interaction,
- graphene/h-BN heterogeneous structures,
- machine-learning interatomic potentials,
- atomistically informed continuum fracture models,
- phase-field/XDEM fracture simulations,
- multiscale transfer of atomistic fracture parameters.

The long-term objective is to connect atomistic simulations with continuum-scale fracture models by extracting physically meaningful mechanical and fracture parameters from MLIP-driven molecular simulations.

---

# 26. Requirements

The workflow requires:

- LAMMPS
- MLIP/MTP interface
- trained machine-learning interatomic potential
- Python 3
- NumPy
- Matplotlib
- OVITO

Optional tools may be used for structure generation and post-processing.

---

# 27. Recommended `.gitignore`

Large simulation outputs should normally not be committed.

```gitignore
# LAMMPS trajectory files
*.lammpstrj
*.dump
dump*

# LAMMPS logs
log.lammps
log.*

# Restart files
*.restart
restart*

# Generated final structures
final_*.data

# Results
results/
outputs/

# Python
__pycache__/
*.pyc

# Scheduler output
slurm-*.out
*.err

# Temporary files
*.tmp
*.bak
```

Small example LAMMPS data files can still be included under

```text
examples/
```

for reproducibility.

---

# 28. Citation

If you use this repository in academic work, please cite the corresponding publication once available.

```bibtex
@article{Kumar_2D_Fracture,
  author  = {Vijay Kumar},
  title   = {Atomistic Fracture Mechanics of Two-Dimensional Materials
             Using Machine-Learning Interatomic Potentials},
  journal = {To be updated},
  year    = {2026}
}
```

---

# 29. Author

**Vijay Kumar**

Research interests:

- Computational materials science
- Two-dimensional materials
- Fracture mechanics
- Molecular dynamics
- Density functional theory
- Machine-learning interatomic potentials
- Graph neural networks
- Multiscale materials modeling

---

## Status

**Active development**

The repository currently contains LAMMPS implementations for tensile,
shear and mixed-mode fracture of cracked 2D materials. Additional
geometries, materials and multiscale coupling approaches are under
development.
