#!/usr/bin/env python3
"""
Cylindrical Wedge Z-Pinch Model: Spent Fuel Shell Position in Zap Energy Blanket
=================================================================================

This script models a 36° (1/10th) cylindrical wedge of the Zap Energy SFS Z-pinch
fusion-fission hybrid blanket.  The wedge extends radially from a small inner
radius (the plasma cavity boundary) to the outer shielding vessel wall at 150 cm,
and spans the full blanket height with physical (vacuum) ceiling and floor
boundaries to capture axial leakage.

The Z-pinch plasma column runs along the central axis (z-axis).  14.1 MeV D-T
fusion neutrons are emitted isotropically from a thin cylindrical shell source
at the inner blanket surface, distributed over the 50 cm pinch length.  The
reflective inner surface approximates the symmetric radial coverage (neutrons
heading inward traverse the plasma and hit blanket on the far side).  Reflective
azimuthal faces enforce the 1/10th rotational symmetry.

An annular shell of homogenized fuel zone material is inserted at varying radial
depths from the inner surface.  The script sweeps the shell position and tracks
key nuclear tallies at each position — identical physics to the sigma-pile model
but in cylindrically correct geometry.

The fuel zone is homogenized from hex-can subassemblies consistent with a
4S-type fast reactor spent fuel design:
    30 vol%  Spent U-10Zr metallic fuel (from CSV depletion inventory, 11.0 g/cm³)
    20 vol%  HT-9 ferritic-martensitic steel cladding (7.87 g/cm³)
    50 vol%  PbLi eutectic coolant (9.49 g/cm³)

Geometry (radial cross-section at a given shell offset d from the inner surface):

     inner                                                    outer
     wall                                                     wall
      |<--- d --->|<-- t -->|<---- R_outer-R_inner-d-t ---->|
      |   PbLi    |  Fuel   |          PbLi                 |
      |           |  shell  |                               |
    r=R_in      r=R_in+d  r=R_in+d+t                    r=R_out
    (refl.)                                              (vacuum)

    Axial:  z = -H/2 (vacuum floor) to z = +H/2 (vacuum ceiling)
    Azimuthal: φ = 0 (reflective) to φ = 36° (reflective)

Tallies tracked:
  1. Tritium breeding rate in PbLi  (Li6(n,t) + Li7(n,nt))
  2. Fission rate in fuel zone shell
  3. (n,gamma) capture rate in fuel zone
  4. (n,2n) reaction rate in PbLi   (Pb neutron multiplication)
  5. Neutron flux spectrum in fuel zone shell (725-group)
  6. Heating (energy deposition) in each region
  7. Neutron leakage (current) through outer wall, ceiling, and floor
  8. Total absorption in fuel zone

Author: Generated for AST 570 coursework
"""

import os
import csv
import re
import json
import numpy as np
try:
    import openmc
except ImportError:
    raise ImportError(
        "OpenMC Python API is required. Install with:\n"
        "  conda install -c conda-forge openmc\n"
        "or build from source: https://docs.openmc.org"
    )

"""
MATPLOTLIB SETTINGS
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Fonts
# try:
#     font_path = './Python/fonts/DIN-Regular.ttf' # DIN-Regular.ttf' # './Python/arial.ttf'
#     fm.fontManager.addfont(font_path)
#     prop = fm.FontProperties(fname=font_path)
#     plt.rcParams['font.family'] = prop.get_name()
# except:
font_path = './figures/fonts/arial.ttf'
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['pdf.fonttype'] = 42

# Ticks
plt.rcParams['xtick.direction']   = 'in'
plt.rcParams['ytick.direction']   = 'in'
plt.rcParams['xtick.major.width'] = 0.6
plt.rcParams['ytick.major.width'] = 0.6
plt.rcParams['xtick.major.size']  = 3.5
plt.rcParams['ytick.major.size']  = 3.5
plt.rcParams['xtick.minor.size']  = 2.0
plt.rcParams['ytick.minor.size']  = 2.0
plt.rcParams['axes.linewidth']    = 0.5
plt.rcParams['grid.color'] = '#DBDBDB'
plt.rcParams['grid.linewidth'] = 0.5

# Font sizes
plt.rcParams['font.size']        =  8
plt.rcParams['axes.titlesize']   =  9
plt.rcParams['axes.labelsize']   =  8
plt.rcParams['xtick.labelsize']  =  7
plt.rcParams['ytick.labelsize']  =  7
plt.rcParams['legend.fontsize']  =  7
plt.rcParams['figure.titlesize'] = 10

# ==============================================================================
# Configuration
# ==============================================================================
CSV_PATH = "uzr_bol_inventory_zpinch.csv"     # Path to spent fuel inventory CSV

# --- Cylindrical wedge geometry ---
R_INNER        = 0.25      # Inner blanket radius [cm] (plasma cavity boundary)
R_OUTER        = 200.0     # Outer shielding vessel radius [cm] (3 m diameter)
HALF_HEIGHT    = 150.0     # Half-height of blanket [cm] (total 150 cm)
WEDGE_ANGLE_DEG = 36.0     # Wedge opening angle [degrees] (1/10th of cylinder)

# --- Source ---
PINCH_HALF_LENGTH = 25.0   # Half-length of the Z-pinch plasma [cm] (50 cm total)
SOURCE_SHELL_DR   = 0.01   # Thickness of source shell at inner surface [cm]

EV_TO_J = 1.602176634e-19   # J per eV
PULSE_HZ = 10.0              # Hz (pulses per second)
N_PER_PULSE = 7.1e18
N_PER_SEC   = N_PER_PULSE * PULSE_HZ

# --- Fuel shell ---
FUEL_SHELL_THICKNESS = 20.0  # Radial thickness of spent fuel annular shell [cm]

# Shell midpoint positions [cm] — radial distance from R_INNER to the
# centre of the 20 cm spent fuel shell.  Sweep covers the middle portion
# of the blanket so there is always PbLi on both sides of the shell.
_BLANKET_DEPTH = R_OUTER - R_INNER  # 
SHELL_MIDPOINTS = [30.0, 50.0, 75, 100.0, 125, 150.0, 175, 189.0] 

# --- Mesh tally ---
MESH_DR   = 0.1                                   # Radial mesh resolution [cm]
N_MESH_R  = int((_BLANKET_DEPTH) / MESH_DR)       # Number of radial bins

# --- Monte Carlo settings ---
BATCHES   = 100
PARTICLES = int(1e3)    # Increase for production runs (e.g. 50000)

# --- PbLi eutectic parameters (Li-17Pb-83 by atom fraction) ---
PBLI_LI_ATOM_FRAC = 0.17
PBLI_PB_ATOM_FRAC = 0.83
PBLI_DENSITY = 9.49                               # g/cm³ at ~500 °C

# --- Spent fuel density — U-10Zr metallic alloy ---
SPENT_FUEL_DENSITY = 11.0                          # g/cm³

# --- HT-9 ferritic-martensitic steel cladding (12Cr-1MoVW) ---
HT9_DENSITY = 7.87                                # g/cm³

# --- Fuel zone volume fractions ---
FUEL_VOL_FRAC    = 0.30                            # 30% metallic fuel slugs
CLAD_VOL_FRAC    = 0.20                            # 20% HT-9 cladding + wire wrap
COOLANT_VOL_FRAC = 0.50                            # 50% PbLi coolant

OUTPUT_DIR = f"uzr_zpinch_results_shell{f'{FUEL_SHELL_THICKNESS:.0f}'.zfill(2)}cm"

# Nuclide groups for per-nuclide mesh tallies (neutron balance breakdown)
KEY_ACTINIDES = ["U235", "U238", "Pu239", "Am241"]
OTHER_ACTINIDES = [
    "Am242", "Am243", "Np237", "Pu238", "Pu240", "Pu241", "Pu242",
    "U234", "U236",
]
ALL_ACTINIDES = KEY_ACTINIDES + OTHER_ACTINIDES
PB_ISOTOPES = ["Pb204", "Pb206", "Pb207", "Pb208"]

# Derived constants
WEDGE_ANGLE_RAD = np.radians(WEDGE_ANGLE_DEG)


# ==============================================================================
# Parse spent fuel inventory from CSV
# ==============================================================================

def parse_inventory(csv_path):
    """
    Read the BOL inventory CSV and return a dict of {openmc_nuclide_name: atom_count}.

    Metastable states (e.g. 'Am242_m1') are folded into their ground-state
    counterparts.  Noble gases (He, Kr, Xe) are skipped — they escape
    metallic fuel.
    """
    inventory = {}
    noble_gas_elements = {"He", "Kr", "Xe"}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_name = row["nuclide"].strip()
            atoms = float(row["atoms"].strip())

            match = re.match(r"([A-Z][a-z]?)", raw_name)
            if not match:
                continue
            element = match.group(1)

            if element in noble_gas_elements:
                continue

            nuclide_name = re.sub(r"_m\d+$", "", raw_name)
            inventory[nuclide_name] = inventory.get(nuclide_name, 0.0) + atoms

    return inventory


def build_spent_fuel_material(inventory):
    """Build an OpenMC Material from the atom inventory."""
    total_atoms = sum(inventory.values())
    fuel = openmc.Material(name="Spent_Fuel_U10Zr")
    fuel.set_density("g/cm3", SPENT_FUEL_DENSITY)

    for nuclide, atoms in inventory.items():
        atom_frac = atoms / total_atoms
        if atom_frac < 1.0e-12:
            continue
        try:
            fuel.add_nuclide(nuclide, atom_frac, percent_type="ao")
        except Exception:
            print(f"  Warning: Nuclide '{nuclide}' not found in library, skipping.")

    return fuel


# ==============================================================================
# Build PbLi eutectic material
# ==============================================================================

def build_pbli_material():
    """Build eutectic PbLi (Li-17Pb-83 by atom %) using natural isotopic abundances."""
    pbli = openmc.Material(name="PbLi_eutectic")
    pbli.set_density("g/cm3", PBLI_DENSITY)

    pbli.add_nuclide("Li6",  PBLI_LI_ATOM_FRAC * 0.0759,  "ao")
    pbli.add_nuclide("Li7",  PBLI_LI_ATOM_FRAC * 0.9241,  "ao")

    pbli.add_nuclide("Pb204", PBLI_PB_ATOM_FRAC * 0.014,  "ao")
    pbli.add_nuclide("Pb206", PBLI_PB_ATOM_FRAC * 0.241,  "ao")
    pbli.add_nuclide("Pb207", PBLI_PB_ATOM_FRAC * 0.221,  "ao")
    pbli.add_nuclide("Pb208", PBLI_PB_ATOM_FRAC * 0.524,  "ao")

    return pbli


# ==============================================================================
# Build HT-9 cladding material
# ==============================================================================

def build_ht9_material():
    """Build HT-9 ferritic-martensitic steel (12Cr-1MoVW)."""
    ht9 = openmc.Material(name="HT9_steel")
    ht9.add_element("Fe", 0.845, percent_type="wo")
    ht9.add_element("Cr", 0.120, percent_type="wo")
    ht9.add_element("Mo", 0.010, percent_type="wo")
    ht9.add_element("W",  0.005, percent_type="wo")
    ht9.add_element("V",  0.003, percent_type="wo")
    ht9.add_element("Mn", 0.006, percent_type="wo")
    ht9.add_element("Ni", 0.005, percent_type="wo")
    ht9.add_element("Si", 0.004, percent_type="wo")
    ht9.add_element("C",  0.002, percent_type="wo")
    ht9.set_density("g/cm3", HT9_DENSITY)
    return ht9


# ==============================================================================
# Build homogenized fuel zone (fuel + cladding + coolant)
# ==============================================================================

def build_fuel_zone_homogenized(fuel_mat, ht9_mat, pbli_mat):
    """
    Homogenize hex-can subassemblies into a single material.

    Volume fractions (wire-wrapped pin bundle in hex duct):
        Fuel slugs:     FUEL_VOL_FRAC    (metallic U-10Zr at smear density)
        HT-9 steel:     CLAD_VOL_FRAC    (cladding + wire wrap + hex duct)
        PbLi coolant:   COOLANT_VOL_FRAC  (flowing through & around assemblies)
    """
    fuel_zone = openmc.Material.mix_materials(
        [fuel_mat, ht9_mat, pbli_mat],
        [FUEL_VOL_FRAC, CLAD_VOL_FRAC, COOLANT_VOL_FRAC],
        percent_type="vo",
        name="Fuel_zone_homogenized"
    )
    return fuel_zone


# ==============================================================================
# Mesh tally I/O helpers
# ==============================================================================

def _save_mesh_csv(path, mesh_data):
    """Write depth-resolved mesh tally data to a CSV file."""
    first_key = list(mesh_data.keys())[0]
    n = len(mesh_data[first_key])
    cols = list(mesh_data.keys())

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for i in range(n):
            row = []
            for c in cols:
                v = mesh_data[c]
                row.append(f"{v[i]:.8e}" if hasattr(v, "__len__") else f"{v:.8e}")
            writer.writerow(row)


def _read_mesh_csv(path):
    """Read a mesh depth-profile CSV back into a dict of numpy arrays."""
    data = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for key in rows[0]:
        try:
            data[key] = np.array([float(r[key]) for r in rows])
        except (ValueError, KeyError):
            pass
    return data


# ==============================================================================
# Build geometry and run for a given fuel shell position
# ==============================================================================

def run_case(shell_midpoint, pbli_mat, fuel_mat, run_index):
    """
    Build and run an OpenMC model for the spent fuel annular shell centred
    at a given radial depth in the blanket.  Pass *shell_midpoint=None*
    for a pure-PbLi baseline (no spent fuel).

    Parameters
    ----------
    shell_midpoint : float or None
        Radial distance [cm] from the inner blanket surface (R_INNER) to
        the *midpoint* of the fuel shell, or None for pure-PbLi reference.
    pbli_mat : openmc.Material
        The PbLi blanket material.
    fuel_mat : openmc.Material
        The homogenized fuel zone material.
    run_index : int
        Index for the parametric sweep (used for directory naming).

    Returns
    -------
    dict : Tally results for this case.
    """

    no_fuel = shell_midpoint is None

    if no_fuel:
        shell_offset = None
        case_dir = os.path.join(OUTPUT_DIR, f"uzr_case_{run_index:03d}_no_fuel")
    else:
        shell_offset = shell_midpoint - FUEL_SHELL_THICKNESS / 2.0
        case_dir = os.path.join(
            OUTPUT_DIR,
            f"uzr_case_{run_index:03d}_mid_{shell_midpoint:.1f}cm"
        )
    os.makedirs(case_dir, exist_ok=True)

    try:
        openmc.reset_auto_ids()
    except AttributeError:
        pass

    # ------------------------------------------------------------------
    # Materials — clone so each case is independent
    # ------------------------------------------------------------------
    pbli_inner = pbli_mat.clone()
    pbli_inner.name = "PbLi_inner"

    if no_fuel:
        materials = openmc.Materials([pbli_inner])
    else:
        pbli_outer = pbli_mat.clone()
        pbli_outer.name = "PbLi_outer"
        fuel = fuel_mat.clone()
        fuel.name = "Fuel_Zone_Homogenized"
        materials = openmc.Materials([pbli_inner, pbli_outer, fuel])

    # ------------------------------------------------------------------
    # Geometry — 36° cylindrical wedge
    # ------------------------------------------------------------------
    # Radial surfaces (cylinders concentric with z-axis)
    cyl_inner = openmc.ZCylinder(r=R_INNER, boundary_type="reflective")
    cyl_outer = openmc.ZCylinder(r=R_OUTER, boundary_type="vacuum")

    # Axial surfaces — physical ceiling and floor (vacuum → captures leakage)
    z_floor   = openmc.ZPlane(z0=-HALF_HEIGHT, boundary_type="vacuum")
    z_ceiling = openmc.ZPlane(z0=+HALF_HEIGHT, boundary_type="vacuum")

    # Azimuthal surfaces — two planes through the z-axis forming the wedge
    # Plane at φ = 0:  the y = 0 plane (XZ plane).  Inside the wedge: y > 0.
    phi_lo = openmc.YPlane(y0=0.0, boundary_type="reflective")

    # Plane at φ = WEDGE_ANGLE:  -sin(θ)·x + cos(θ)·y = 0
    # Inside the wedge: -sin(θ)·x + cos(θ)·y < 0  (negative half-space)
    sin_w = np.sin(WEDGE_ANGLE_RAD)
    cos_w = np.cos(WEDGE_ANGLE_RAD)
    phi_hi = openmc.Plane(a=-sin_w, b=cos_w, c=0.0, d=0.0,
                          boundary_type="reflective")

    # Common region components
    wedge  = +phi_lo & -phi_hi            # azimuthal wedge
    axial  = +z_floor & -z_ceiling        # between floor and ceiling
    radial = +cyl_inner & -cyl_outer      # between inner and outer cylinders
    common = wedge & axial

    cells = []

    if no_fuel:
        region_all = radial & common
        cell_all = openmc.Cell(name="PbLi_full", fill=pbli_inner,
                               region=region_all)
        cells.append(cell_all)
    else:
        # Fuel shell radii
        r_fuel_in  = R_INNER + shell_offset
        r_fuel_out = R_INNER + shell_offset + FUEL_SHELL_THICKNESS
        cyl_fuel_in  = openmc.ZCylinder(r=r_fuel_in)
        cyl_fuel_out = openmc.ZCylinder(r=r_fuel_out)

        has_inner_pbli = shell_offset > 0.0
        has_outer_pbli = r_fuel_out < R_OUTER

        # Inner PbLi region (between inner wall and fuel shell)
        if has_inner_pbli:
            region_inner = +cyl_inner & -cyl_fuel_in & common
            cell_inner = openmc.Cell(name="PbLi_inner", fill=pbli_inner,
                                     region=region_inner)
            cells.append(cell_inner)

        # Spent fuel annular shell
        region_fuel = +cyl_fuel_in & -cyl_fuel_out & common
        cell_fuel = openmc.Cell(name="Fuel_Zone_Shell", fill=fuel,
                                region=region_fuel)
        cells.append(cell_fuel)

        # Outer PbLi region (between fuel shell and outer wall)
        if has_outer_pbli:
            region_outer = +cyl_fuel_out & -cyl_outer & common
            cell_outer = openmc.Cell(name="PbLi_outer", fill=pbli_outer,
                                     region=region_outer)
            cells.append(cell_outer)

    universe = openmc.Universe(cells=cells)
    geometry = openmc.Geometry(universe)

    # ------------------------------------------------------------------
    # Source: 14.1 MeV D-T neutrons from Z-pinch plasma column
    # ------------------------------------------------------------------
    # Thin cylindrical shell at the inner blanket surface, distributed
    # along the 50 cm pinch length, isotropic angular distribution.
    # The reflective inner surface handles "through-plasma" neutrons.
    source = openmc.IndependentSource()
    source.space = openmc.stats.CylindricalIndependent(
        r=openmc.stats.Uniform(R_INNER, R_INNER + SOURCE_SHELL_DR),
        phi=openmc.stats.Uniform(0.0, WEDGE_ANGLE_RAD),
        z=openmc.stats.Uniform(-PINCH_HALF_LENGTH, +PINCH_HALF_LENGTH),
        origin=(0.0, 0.0, 0.0),
    )
    source.angle = openmc.stats.Isotropic()
    source.energy = openmc.stats.Discrete([14.1e6], [1.0])

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.batches = BATCHES
    settings.inactive = 0
    settings.particles = PARTICLES
    settings.source = source

    # ------------------------------------------------------------------
    # Tallies
    # ------------------------------------------------------------------
    tallies = openmc.Tallies()

    # --- Tally 1: Tritium Breeding Rate in ALL PbLi ---
    if no_fuel:
        pbli_filter_mats = [pbli_inner]
    else:
        pbli_filter_mats = []
        if has_inner_pbli:
            pbli_filter_mats.append(pbli_inner)
        if has_outer_pbli:
            pbli_filter_mats.append(pbli_outer)

    if pbli_filter_mats:
        tbr_tally = openmc.Tally(name="TBR")
        tbr_tally.filters = [openmc.MaterialFilter(pbli_filter_mats)]
        tbr_tally.scores = ["(n,Xt)"]
        tallies.append(tbr_tally)

    # --- Tally 2: Fission rate in spent fuel ---
    if not no_fuel:
        fission_tally = openmc.Tally(name="fission_rate")
        fission_tally.filters = [openmc.MaterialFilter([fuel])]
        fission_tally.scores = ["fission"]
        tallies.append(fission_tally)

    # --- Tally 3: Radiative capture in spent fuel ---
    if not no_fuel:
        capture_tally = openmc.Tally(name="capture_rate")
        capture_tally.filters = [openmc.MaterialFilter([fuel])]
        capture_tally.scores = ["(n,gamma)"]
        tallies.append(capture_tally)

    # --- Tally 4: (n,2n) in PbLi ---
    if pbli_filter_mats:
        n2n_tally = openmc.Tally(name="n2n_rate")
        n2n_tally.filters = [openmc.MaterialFilter(pbli_filter_mats)]
        n2n_tally.scores = ["(n,2n)"]
        tallies.append(n2n_tally)

    # --- Tally 5: Neutron flux spectrum in spent fuel (725-group) ---
    energy_bins = np.logspace(np.log10(1e-5), np.log10(20e6), 726)
    if not no_fuel:
        energy_filter = openmc.EnergyFilter(energy_bins)
        spectrum_tally = openmc.Tally(name="fuel_spectrum")
        spectrum_tally.filters = [openmc.MaterialFilter([fuel]), energy_filter]
        spectrum_tally.scores = ["flux"]
        tallies.append(spectrum_tally)

    # --- Tally 6: Heating in each region ---
    all_mats = pbli_filter_mats + ([fuel] if not no_fuel else [])
    heating_tally = openmc.Tally(name="heating_local")
    heating_tally.filters = [openmc.MaterialFilter(all_mats)]
    heating_tally.scores = ["heating-local"]
    tallies.append(heating_tally)

    fisq_tally = openmc.Tally(name="fission_q_recoverable")
    fisq_tally.filters = [openmc.MaterialFilter(all_mats)]
    fisq_tally.scores = ["fission-q-recoverable"]
    tallies.append(fisq_tally)

    # --- Tally 7: Neutron leakage through outer wall, ceiling, floor ---
    leak_surfaces = [cyl_outer, z_ceiling, z_floor]
    leakage_tally = openmc.Tally(name="leakage")
    leakage_tally.filters = [openmc.SurfaceFilter(leak_surfaces)]
    leakage_tally.scores = ["current"]
    tallies.append(leakage_tally)

    # --- Tally 8: Absorption in spent fuel ---
    if not no_fuel:
        abs_tally = openmc.Tally(name="absorption_fuel")
        abs_tally.filters = [openmc.MaterialFilter([fuel])]
        abs_tally.scores = ["absorption"]
        tallies.append(abs_tally)

    # ------------------------------------------------------------------
    # Mesh Tally: radial depth-resolved profiles (CylindricalMesh)
    # ------------------------------------------------------------------
    # Single azimuthal bin (full wedge) and single axial bin (full height).
    # N_MESH_R radial bins at MESH_DR resolution give a radial profile.
    mesh = openmc.CylindricalMesh(
        r_grid=np.linspace(R_INNER, R_OUTER, N_MESH_R + 1),
        z_grid=np.array([-HALF_HEIGHT, +HALF_HEIGHT]),
        phi_grid=np.array([0.0, WEDGE_ANGLE_RAD]),
    )

    mesh_filter = openmc.MeshFilter(mesh)

    MESH_SCORES = [
        "(n,Xt)",       # 1. Tritium breeding rate
        "fission",      # 2. Fission rate
        "(n,gamma)",    # 3. Radiative capture
        "(n,2n)",       # 4. Neutron multiplication (Pb)
        "flux",         # 5. Neutron flux (depth profile)
        "heating",      # 6. Energy deposition [eV/src]
        "absorption",   # 7. Total absorption
        "nu-fission",   # 8. Fission neutron production
    ]

    mesh_tally = openmc.Tally(name="mesh_depth_profile")
    mesh_tally.filters = [mesh_filter]
    mesh_tally.scores = MESH_SCORES
    tallies.append(mesh_tally)

    # Surface-current mesh tally (radial current at each depth)
    mesh_surface_filter = openmc.MeshSurfaceFilter(mesh)
    current_mesh_tally = openmc.Tally(name="mesh_current_profile")
    current_mesh_tally.filters = [mesh_surface_filter]
    current_mesh_tally.scores = ["current"]
    tallies.append(current_mesh_tally)

    # ------------------------------------------------------------------
    # Per-nuclide mesh tallies for neutron balance breakdown
    # ------------------------------------------------------------------
    if not no_fuel:
        nuc_key_tally = openmc.Tally(name="mesh_key_actinides")
        nuc_key_tally.filters = [mesh_filter]
        nuc_key_tally.nuclides = KEY_ACTINIDES
        nuc_key_tally.scores = [
            "fission", "nu-fission", "(n,gamma)", "(n,2n)", "(n,3n)"
        ]
        tallies.append(nuc_key_tally)

        nuc_other_tally = openmc.Tally(name="mesh_other_actinides")
        nuc_other_tally.filters = [mesh_filter]
        nuc_other_tally.nuclides = OTHER_ACTINIDES
        nuc_other_tally.scores = ["fission", "nu-fission", "(n,gamma)"]
        tallies.append(nuc_other_tally)

    pb_n2n_tally = openmc.Tally(name="mesh_pb_n2n")
    pb_n2n_tally.filters = [mesh_filter]
    pb_n2n_tally.nuclides = PB_ISOTOPES
    pb_n2n_tally.scores = ["(n,2n)"]
    tallies.append(pb_n2n_tally)

    # ------------------------------------------------------------------
    # Build and run the model
    # ------------------------------------------------------------------
    model = openmc.Model(geometry=geometry, materials=materials,
                         settings=settings, tallies=tallies)

    original_dir = os.getcwd()
    os.chdir(case_dir)

    try:
        sp_filename = f"statepoint.{BATCHES}.h5"

        if os.path.isfile(sp_filename):
            print(f"  Statepoint '{sp_filename}' already exists — skipping run.")
        else:
            model.export_to_model_xml()
            openmc.run(output=True)

        # Extract results
        sp_file = openmc.StatePoint(sp_filename)
        results = {
            "shell_midpoint_cm": shell_midpoint,
            "shell_offset_cm":  shell_offset,
            "no_fuel":          no_fuel,
        }

        # TBR
        if pbli_filter_mats:
            t = sp_file.get_tally(name="TBR")
            results["TBR_mean"] = float(t.mean.sum())
            results["TBR_std"]  = float(np.sqrt((t.std_dev**2).sum()))

        # Fission rate
        if not no_fuel:
            t = sp_file.get_tally(name="fission_rate")
            results["fission_mean"] = float(t.mean.sum())
            results["fission_std"]  = float(np.sqrt((t.std_dev**2).sum()))

            t = sp_file.get_tally(name="capture_rate")
            results["capture_mean"] = float(t.mean.sum())
            results["capture_std"]  = float(np.sqrt((t.std_dev**2).sum()))
        else:
            results["fission_mean"] = results["fission_std"] = 0.0
            results["capture_mean"] = results["capture_std"] = 0.0

        # (n,2n) rate
        if pbli_filter_mats:
            t = sp_file.get_tally(name="n2n_rate")
            results["n2n_mean"] = float(t.mean.sum())
            results["n2n_std"]  = float(np.sqrt((t.std_dev**2).sum()))

        # Neutron spectrum in fuel
        if not no_fuel:
            t = sp_file.get_tally(name="fuel_spectrum")
            results["fuel_spectrum_mean"] = t.mean.flatten().tolist()
            results["fuel_spectrum_std"]  = t.std_dev.flatten().tolist()
        results["energy_bins_eV"] = energy_bins.tolist()

        # Heating
        t = sp_file.get_tally(name="heating_local")
        results["heating_eV_per_sp"] = { m.name: float(t.mean[i][0][0]) for i, m in enumerate(all_mats) }
        results["heating_W"] = { name: val * EV_TO_J * N_PER_SEC for name, val in results["heating_eV_per_sp"].items() }

        # Fission Q recoverable
        t = sp_file.get_tally(name="fission_q_recoverable")
        results["fisq_eV_per_sp"] = { m.name: float(t.mean[i][0][0]) for i, m in enumerate(all_mats) }
        results["fisq_W"] = { name: val * EV_TO_J * N_PER_SEC for name, val in results["fisq_eV_per_sp"].items() }

        # Leakage — itemised by surface (outer wall, ceiling, floor)
        t = sp_file.get_tally(name="leakage")
        leak_mean = t.mean.flatten()
        leak_std  = t.std_dev.flatten()
        results["leakage_outer_mean"]   = float(leak_mean[0])
        results["leakage_ceiling_mean"] = float(leak_mean[1])
        results["leakage_floor_mean"]   = float(leak_mean[2])
        results["leakage_mean"]  = float(leak_mean.sum())
        results["leakage_std"]   = float(np.sqrt((leak_std**2).sum()))
        results["leakage_outer_std"]   = float(leak_std[0])
        results["leakage_ceiling_std"] = float(leak_std[1])
        results["leakage_floor_std"]   = float(leak_std[2])

        # Absorption in fuel
        if not no_fuel:
            t = sp_file.get_tally(name="absorption_fuel")
            results["absorption_mean"] = float(t.mean.sum())
            results["absorption_std"]  = float(np.sqrt((t.std_dev**2).sum()))
        else:
            results["absorption_mean"] = results["absorption_std"] = 0.0

        # -----------------------------------------------------------
        # Extract depth-resolved mesh tallies (radial profiles)
        # -----------------------------------------------------------
        r_edges   = np.linspace(R_INNER, R_OUTER, N_MESH_R + 1)
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        depth_centers = r_centers - R_INNER     # depth from inner wall

        mt = sp_file.get_tally(name="mesh_depth_profile")
        n_scores = len(MESH_SCORES)
        mt_mean = mt.mean.flatten().reshape(N_MESH_R, n_scores)
        mt_std  = mt.std_dev.flatten().reshape(N_MESH_R, n_scores)

        mesh_data = {
            "r_cm":     r_centers,
            "depth_cm": depth_centers,
        }
        for si, score_name in enumerate(MESH_SCORES):
            col = (score_name.replace("(", "").replace(")", "")
                   .replace(",", "").replace("-", "_"))
            mesh_data[f"{col}_mean"] = mt_mean[:, si]
            mesh_data[f"{col}_std"]  = mt_std[:, si]

        # Surface-current mesh tally (radial outward current)
        ct = sp_file.get_tally(name="mesh_current_profile")
        ct_mean = ct.mean.flatten()
        ct_std  = ct.std_dev.flatten()
        # For CylindricalMesh [Nr,1,1], each voxel has 6 surfaces:
        #   0: r_in,  1: r_out,  2: phi_lo,  3: phi_hi,  4: z_bot,  5: z_top
        # The outward (+r) surface of voxel i is at index 6*i + 1.
        n_surf_per_vox = 6
        r_out_indices = np.arange(N_MESH_R) * n_surf_per_vox + 1
        if r_out_indices[-1] < len(ct_mean):
            mesh_data["current_r_mean"] = ct_mean[r_out_indices]
            mesh_data["current_r_std"]  = ct_std[r_out_indices]
        else:
            np.save("mesh_current_raw_mean.npy", ct_mean)
            np.save("mesh_current_raw_std.npy",  ct_std)
            print("  Note: MeshSurface current saved as raw .npy "
                  "for manual slicing.")

        mesh_csv_path = "uzr_mesh_depth_profiles.csv"
        _save_mesh_csv(mesh_csv_path, mesh_data)

        # ----------------------------------------------------------
        # k-inf from mesh tally (nu-fission / absorption over fuel shell)
        # ----------------------------------------------------------
        if not no_fuel:
            fuel_mask = ((depth_centers >= shell_offset) &
                         (depth_centers < shell_offset + FUEL_SHELL_THICKNESS))
            nuf_sum = mesh_data["nu_fission_mean"][fuel_mask].sum()
            abs_sum = mesh_data["absorption_mean"][fuel_mask].sum()
            nuf_std_sum = np.sqrt((mesh_data["nu_fission_std"][fuel_mask]**2).sum())
            abs_std_sum = np.sqrt((mesh_data["absorption_std"][fuel_mask]**2).sum())
            if abs_sum > 0:
                kinf = nuf_sum / abs_sum
                rel_nuf = nuf_std_sum / nuf_sum if nuf_sum > 0 else 0.0
                rel_abs = abs_std_sum / abs_sum if abs_sum > 0 else 0.0
                kinf_std = kinf * np.sqrt(rel_nuf**2 + rel_abs**2)
            else:
                kinf = kinf_std = 0.0
            results["k_inf"]     = kinf
            results["k_inf_std"] = kinf_std
        else:
            results["k_inf"] = results["k_inf_std"] = 0.0

        # ----------------------------------------------------------
        # Per-nuclide mesh tallies for neutron balance
        # ----------------------------------------------------------
        nbal = {
            "r_cm":     r_centers,
            "depth_cm": depth_centers,
        }

        pb_t = sp_file.get_tally(name="mesh_pb_n2n")
        pb_data = pb_t.mean.flatten().reshape(
            N_MESH_R, len(PB_ISOTOPES), 1)
        nbal["pb_n2n"] = pb_data[:, :, 0].sum(axis=1)

        if not no_fuel:
            ka_t = sp_file.get_tally(name="mesh_key_actinides")
            ka_scores = [
                "fission", "nu-fission", "(n,gamma)", "(n,2n)", "(n,3n)"
            ]
            ka = ka_t.mean.flatten().reshape(
                N_MESH_R, len(KEY_ACTINIDES), len(ka_scores))
            for ni, nuc in enumerate(KEY_ACTINIDES):
                nbal[f"{nuc}_fission"]    = ka[:, ni, 0]
                nbal[f"{nuc}_nu_fission"] = ka[:, ni, 1]
                nbal[f"{nuc}_ngamma"]     = ka[:, ni, 2]
                nbal[f"{nuc}_n2n"]        = ka[:, ni, 3]
                nbal[f"{nuc}_n3n"]        = ka[:, ni, 4]

            oa_t = sp_file.get_tally(name="mesh_other_actinides")
            oa_scores = ["fission", "nu-fission", "(n,gamma)"]
            oa = oa_t.mean.flatten().reshape(
                N_MESH_R, len(OTHER_ACTINIDES), len(oa_scores))
            nbal["other_act_fission"]    = oa[:, :, 0].sum(axis=1)
            nbal["other_act_nu_fission"] = oa[:, :, 1].sum(axis=1)
            nbal["other_act_ngamma"]     = oa[:, :, 2].sum(axis=1)

        nbal["nXt"] = mt_mean[:, 0]  # score index 0 = (n,Xt)

        nbal_csv_path = "uzr_neutron_balance_depth.csv"
        _save_mesh_csv(nbal_csv_path, nbal)
        results["nbal_csv"] = os.path.join(case_dir, nbal_csv_path)

        results["mesh_csv"] = os.path.join(case_dir, mesh_csv_path)

        sp_file.close()

    finally:
        os.chdir(original_dir)

    return results


# ==============================================================================
# Post-processing: plotting
# ==============================================================================

def plot_results(all_results):
    """Generate summary plots from the parametric sweep."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots.")
        return

    fuel_only = [r for r in all_results if not r.get("no_fuel", False)]
    if not fuel_only:
        return

    midpoints    = [r["shell_midpoint_cm"] for r in fuel_only]
    tbr_mean     = [r.get("TBR_mean", 0.0) for r in fuel_only]
    tbr_std      = [r.get("TBR_std", 0.0)  for r in fuel_only]
    fiss_mean    = [r["fission_mean"]       for r in fuel_only]
    fiss_std     = [r["fission_std"]        for r in fuel_only]
    cap_mean     = [r["capture_mean"]       for r in fuel_only]
    cap_std      = [r["capture_std"]        for r in fuel_only]
    n2n_mean     = [r.get("n2n_mean", 0.0)  for r in fuel_only]
    n2n_std      = [r.get("n2n_std", 0.0)   for r in fuel_only]
    leak_mean    = [r["leakage_mean"]       for r in fuel_only]
    leak_std     = [r["leakage_std"]        for r in fuel_only]
    abs_mean     = [r["absorption_mean"]    for r in fuel_only]
    abs_std      = [r["absorption_std"]     for r in fuel_only]

    fig, axes = plt.subplots(3, 2, figsize=(14, 16), constrained_layout=True)
    fig.suptitle(
        "Z-Pinch Cylindrical Wedge: Fuel Shell Position Sensitivity\n"
        f"(36° wedge, R_in={R_INNER} cm, R_out={R_OUTER} cm, "
        f"H={2*HALF_HEIGHT:.0f} cm)",
        fontsize=13, fontweight="bold"
    )

    xlabel = "Fuel shell midpoint — radial depth from inner wall [cm]"

    ax = axes[0, 0]
    ax.errorbar(midpoints, tbr_mean, yerr=tbr_std, fmt="o-", capsize=3,
                color="tab:blue")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Tritium Production Rate\n[per source neutron]")
    ax.set_title("(a) Tritium Breeding Rate in PbLi")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.errorbar(midpoints, fiss_mean, yerr=fiss_std, fmt="s-", capsize=3,
                color="tab:red")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fission Rate\n[per source neutron]")
    ax.set_title("(b) Fission Rate in Spent Fuel")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.errorbar(midpoints, cap_mean, yerr=cap_std, fmt="^-", capsize=3,
                color="tab:green")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("(n,γ) Capture Rate\n[per source neutron]")
    ax.set_title("(c) Radiative Capture in Spent Fuel")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.errorbar(midpoints, n2n_mean, yerr=n2n_std, fmt="D-", capsize=3,
                color="tab:orange")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("(n,2n) Rate in PbLi\n[per source neutron]")
    ax.set_title("(d) Pb Neutron Multiplication")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.errorbar(midpoints, leak_mean, yerr=leak_std, fmt="v-", capsize=3,
                color="tab:purple")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Total Neutron Leakage\n[per source neutron]")
    ax.set_title("(e) Leakage (outer wall + ceiling + floor)")
    ax.grid(True, alpha=0.3)

    # Stacked leakage breakdown
    ax = axes[2, 1]
    leak_outer   = [r["leakage_outer_mean"]   for r in fuel_only]
    leak_ceil    = [r["leakage_ceiling_mean"]  for r in fuel_only]
    leak_floor   = [r["leakage_floor_mean"]    for r in fuel_only]
    ax.plot(midpoints, leak_outer, "o-", color="tab:red",    label="Outer wall")
    ax.plot(midpoints, leak_ceil,  "s-", color="tab:blue",   label="Ceiling")
    ax.plot(midpoints, leak_floor, "^-", color="tab:green",  label="Floor")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Neutron Leakage\n[per source neutron]")
    ax.set_title("(f) Leakage Breakdown by Surface")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(OUTPUT_DIR, "uzr_zpinch_sensitivity.png")
    fig.savefig(plot_path, dpi=200)
    print(f"\nPlots saved to: {plot_path}")
    plt.close(fig)

    # --- Neutron spectra at select positions ---
    if fuel_only:
        fig2, ax2 = plt.subplots(figsize=(10, 6), constrained_layout=True)
        indices_to_plot = np.linspace(
            0, len(fuel_only) - 1, min(6, len(fuel_only)), dtype=int)
        cmap = plt.cm.viridis(np.linspace(0, 1, len(indices_to_plot)))

        for color_idx, i in enumerate(indices_to_plot):
            r = fuel_only[i]
            e_bins = np.array(r["energy_bins_eV"])
            e_mid  = 0.5 * (e_bins[:-1] + e_bins[1:])
            lethargy_width = np.log(e_bins[1:] / e_bins[:-1])
            flux = np.array(r["fuel_spectrum_mean"])
            flux_per_lethargy = flux / lethargy_width
            ax2.loglog(e_mid, flux_per_lethargy, color=cmap[color_idx],
                       label=f"depth = {r['shell_midpoint_cm']:.0f} cm",
                       alpha=0.8)

        ax2.set_xlabel("Neutron Energy [eV]")
        ax2.set_ylabel("Flux per unit lethargy [n/src/lethargy]")
        ax2.set_title("Neutron Spectrum in Fuel Shell vs. Radial Position")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, which="both")

        spec_path = os.path.join(
            OUTPUT_DIR, "uzr_zpinch_fuel_spectra_vs_position.png")
        fig2.savefig(spec_path, dpi=200)
        print(f"Spectra plot saved to: {spec_path}")
        plt.close(fig2)


def plot_mesh_profiles(all_results):
    """Generate radial depth-resolved mesh tally plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fuel_results = [r for r in all_results if not r.get("no_fuel", False)]
    if not fuel_results:
        return
    indices = np.linspace(
        0, len(fuel_results) - 1, min(6, len(fuel_results)), dtype=int)
    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(indices)))

    blanket_depth = R_OUTER - R_INNER

    score_panels = [
        ("nXt_mean",        "Tritium production (n,Xt)",    "[per src / mesh bin]"),
        ("fission_mean",    "Fission rate",                 "[per src / mesh bin]"),
        ("ngamma_mean",     "Radiative capture (n,γ)",      "[per src / mesh bin]"),
        ("n2n_mean",        "Pb neutron mult. (n,2n)",      "[per src / mesh bin]"),
        ("flux_mean",       "Neutron flux",                 "[n·cm / src / mesh bin]"),
        ("heating_mean",    "Heating",                      "[eV / src / mesh bin]"),
        ("absorption_mean", "Total absorption",             "[per src / mesh bin]"),
        ("nu_fission_mean", "ν-Fission neutron production", "[per src / mesh bin]"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(15, 20), constrained_layout=True)
    fig.suptitle(
        "Radial Depth-Resolved Mesh Tallies — Z-Pinch Cylindrical Wedge",
        fontsize=14, fontweight="bold",
    )

    for panel_idx, (col, title, ylabel) in enumerate(score_panels):
        ax = axes.flat[panel_idx]
        for ci, case_i in enumerate(indices):
            r = fuel_results[case_i]
            csv_path = r.get("mesh_csv")
            if csv_path is None or not os.path.exists(csv_path):
                continue

            data = _read_mesh_csv(csv_path)
            depth = data.get("depth_cm")
            y = data.get(col)
            if depth is None or y is None:
                continue

            offset = r["shell_offset_cm"]
            midpt  = r["shell_midpoint_cm"]
            ax.semilogy(depth, y, color=cmap[ci], alpha=0.75, linewidth=0.8,
                        label=f"depth = {midpt:.0f} cm")

            ax.axvspan(offset, offset + FUEL_SHELL_THICKNESS,
                       color=cmap[ci], alpha=0.08)

        ax.set_xlabel("Radial depth from inner wall [cm]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best", ncol=2)
        ax.grid(True, alpha=0.25, which="both")
        ax.set_xlim(0, blanket_depth)

    mesh_plot_path = os.path.join(
        OUTPUT_DIR, "uzr_zpinch_mesh_depth_profiles.png")
    fig.savefig(mesh_plot_path, dpi=200)
    print(f"Mesh depth profile plots saved to: {mesh_plot_path}")
    plt.close(fig)

    # --- Radial current profile ---
    fig3, ax3 = plt.subplots(figsize=(10, 5), constrained_layout=True)
    found_current = False
    for ci, case_i in enumerate(indices):
        r = fuel_results[case_i]
        csv_path = r.get("mesh_csv")
        if csv_path is None or not os.path.exists(csv_path):
            continue
        data = _read_mesh_csv(csv_path)
        depth = data.get("depth_cm")
        j = data.get("current_r_mean")
        if depth is None or j is None:
            continue
        found_current = True
        offset = r["shell_offset_cm"]
        midpt  = r["shell_midpoint_cm"]
        ax3.semilogy(depth, j, color=cmap[ci], alpha=0.75, linewidth=0.8,
                     label=f"depth = {midpt:.0f} cm")
        ax3.axvspan(offset, offset + FUEL_SHELL_THICKNESS,
                    color=cmap[ci], alpha=0.08)

    if found_current:
        ax3.set_xlabel("Radial depth from inner wall [cm]")
        ax3.set_ylabel("Net outward radial current [per src / surface]")
        ax3.set_title("Neutron Current (radial) Depth Profile")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.25, which="both")
        ax3.set_xlim(0, blanket_depth)
        current_path = os.path.join(
            OUTPUT_DIR, "uzr_zpinch_mesh_r_current_profile.png")
        fig3.savefig(current_path, dpi=200)
        print(f"Radial current profile plot saved to: {current_path}")
    plt.close(fig3)


def plot_neutron_balance(all_results):
    """Plot the depth-resolved neutron balance (gains and losses per bin)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fuel_results = [r for r in all_results if not r.get("no_fuel", False)]
    if not fuel_results:
        return

    n_cases = len(fuel_results)
    indices = np.linspace(0, n_cases - 1, n_cases, dtype=int)
    blanket_depth = R_OUTER - R_INNER

    for ci in indices:
        r = fuel_results[ci]
        csv_path = r.get("nbal_csv")
        if csv_path is None or not os.path.exists(csv_path):
            continue

        data = _read_mesh_csv(csv_path)
        depth = data.get("depth_cm")
        if depth is None:
            continue

        midpt  = r["shell_midpoint_cm"]
        offset = r["shell_offset_cm"]

        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

        # --- GAINS (positive) ---
        pb_n2n = data.get("pb_n2n", np.zeros_like(depth))
        ax.fill_between(depth, 0, pb_n2n, alpha=0.3, color="tab:blue",
                        label="Pb (n,2n) gain")
        ax.plot(depth, pb_n2n, color="tab:blue", linewidth=0.6)

        for nuc, color in [("U235", "tab:red"), ("U238", "tab:orange"),
                           ("Pu239", "tab:green")]:
            fiss = data.get(f"{nuc}_fission", np.zeros_like(depth))
            nu_f = data.get(f"{nuc}_nu_fission", np.zeros_like(depth))
            net = nu_f - fiss
            ax.plot(depth, net, color=color, linewidth=1.2,
                    label=f"{nuc} fission net gain")

        am_fiss = data.get("Am241_fission", np.zeros_like(depth))
        am_nuf  = data.get("Am241_nu_fission", np.zeros_like(depth))
        am_net  = am_nuf - am_fiss
        ax.plot(depth, am_net, color="tab:pink", linewidth=1.2,
                label="Am241 fission net gain")

        am_n2n = data.get("Am241_n2n", np.zeros_like(depth))
        if am_n2n.max() > 1e-20:
            ax.plot(depth, am_n2n, color="tab:pink", linewidth=1.0,
                    linestyle="--", label="Am241 (n,2n) gain")

        am_n3n = data.get("Am241_n3n", np.zeros_like(depth))
        if am_n3n.max() > 1e-20:
            ax.plot(depth, 2.0 * am_n3n, color="tab:pink", linewidth=1.0,
                    linestyle=":", label="Am241 (n,3n) gain (\u00d72)")

        oa_fiss = data.get("other_act_fission", np.zeros_like(depth))
        oa_nuf  = data.get("other_act_nu_fission", np.zeros_like(depth))
        oa_net  = oa_nuf - oa_fiss
        if oa_net.max() > 1e-15:
            ax.plot(depth, oa_net, color="tab:olive", linewidth=1.0,
                    linestyle="--", label="Other actinide fission net gain")

        # --- LOSSES (negative) ---
        nXt = data.get("nXt", np.zeros_like(depth))
        ax.fill_between(depth, 0, -nXt, alpha=0.2, color="tab:purple",
                        label="(n,Xt) tritium loss")
        ax.plot(depth, -nXt, color="tab:purple", linewidth=0.6)

        act_cap = np.zeros_like(depth)
        for nuc in KEY_ACTINIDES:
            act_cap += data.get(f"{nuc}_ngamma", np.zeros_like(depth))
        act_cap += data.get("other_act_ngamma", np.zeros_like(depth))
        ax.plot(depth, -act_cap, color="tab:brown", linewidth=1.2,
                label="Actinide (n,\u03b3) loss")

        ax.axvspan(offset, offset + FUEL_SHELL_THICKNESS, color="gray",
                   alpha=0.15, label="Spent fuel shell")

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Radial depth from inner wall [cm]")
        ax.set_ylabel(
            "Net neutrons added (+) or lost (\u2212) per source neutron per bin"
        )
        ax.set_title(
            f"Neutron balance vs radial depth \u2014 "
            f"fuel shell midpoint = {midpt:.0f} cm"
        )
        ax.legend(fontsize=8, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, blanket_depth)

        plot_path = os.path.join(
            OUTPUT_DIR, f"uzr_zpinch_nbal_mid{midpt:.0f}cm.png")
        fig.savefig(plot_path, dpi=200)
        print(f"Neutron balance plot saved to: {plot_path}")
        plt.close(fig)


def plot_cumulative_neutron_balance(all_results):
    """Plot cumulative neutron balance vs radial depth."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fuel_results = [r for r in all_results if not r.get("no_fuel", False)]
    if not fuel_results:
        return

    n_cases = len(fuel_results)
    indices = np.linspace(0, n_cases - 1, n_cases, dtype=int)
    blanket_depth = R_OUTER - R_INNER

    for ci in indices:
        r = fuel_results[ci]
        csv_path = r.get("nbal_csv")
        if csv_path is None or not os.path.exists(csv_path):
            continue

        data = _read_mesh_csv(csv_path)
        depth = data.get("depth_cm")
        if depth is None:
            continue

        midpt  = r["shell_midpoint_cm"]
        offset = r["shell_offset_cm"]

        fig, ax = plt.subplots(figsize=(6.5, 3.0), constrained_layout=True)

        # --- Cumulative GAINS (positive) ---
        pb_n2n = np.cumsum(data.get("pb_n2n", np.zeros_like(depth)))
        ax.plot(depth, pb_n2n, color="tab:blue", linewidth=0.75,
                label="Pb (n,2n) gain")
        # ax.fill_between(depth, 0, pb_n2n, alpha=0.15, color="tab:blue")

        for nuc, color, ls in [("U235", "tab:red", "-"),
                                ("U238", "tab:orange", "-"),
                                ("Pu239", "tab:green", "-")]:
            fiss = data.get(f"{nuc}_fission", np.zeros_like(depth))
            nu_f = data.get(f"{nuc}_nu_fission", np.zeros_like(depth))
            net = np.cumsum(nu_f - fiss)
            ax.plot(depth, net, color=color, linewidth=0.75, linestyle=ls,
                    label=f"{nuc} fission net gain")

        am_fiss = data.get("Am241_fission", np.zeros_like(depth))
        am_nuf  = data.get("Am241_nu_fission", np.zeros_like(depth))
        am_net  = np.cumsum(am_nuf - am_fiss)
        ax.plot(depth, am_net, color="tab:pink", linewidth=0.75,
                label="Am241 fission net gain")

        am_n2n_raw = data.get("Am241_n2n", np.zeros_like(depth))
        am_n2n = np.cumsum(am_n2n_raw)
        if am_n2n[-1] > 1e-20:
            ax.plot(depth, am_n2n, color="tab:pink", linewidth=0.75,
                    linestyle="--", label="Am241 (n,2n) gain")

        am_n3n_raw = data.get("Am241_n3n", np.zeros_like(depth))
        am_n3n = np.cumsum(2.0 * am_n3n_raw)
        if am_n3n[-1] > 1e-20:
            ax.plot(depth, am_n3n, color="tab:pink", linewidth=0.75,
                    linestyle=":", label="Am241 (n,3n) gain (\u00d72)")

        oa_fiss = data.get("other_act_fission", np.zeros_like(depth))
        oa_nuf  = data.get("other_act_nu_fission", np.zeros_like(depth))
        oa_net  = np.cumsum(oa_nuf - oa_fiss)
        if np.abs(oa_net).max() > 1e-15:
            ax.plot(depth, oa_net, color="tab:olive", linewidth=0.75,
                    linestyle="--", label="Other actinide fission net")

        # --- Cumulative LOSSES (negative) ---
        nXt = np.cumsum(data.get("nXt", np.zeros_like(depth)))
        ax.plot(depth, -nXt, color="tab:purple", linewidth=0.75,
                label="(n,Xt) tritium loss")
        # ax.fill_between(depth, 0, -nXt, alpha=0.1, color="tab:purple")

        act_cap_raw = np.zeros_like(depth)
        for nuc in KEY_ACTINIDES:
            act_cap_raw += data.get(f"{nuc}_ngamma", np.zeros_like(depth))
        act_cap_raw += data.get("other_act_ngamma", np.zeros_like(depth))
        act_cap = np.cumsum(act_cap_raw)
        ax.plot(depth, -act_cap, color="tab:brown", linewidth=0.75,
                label="Actinide (n,\u03b3) loss")

        # --- Total cumulative budget ---
        total = pb_n2n + np.cumsum(
            (data.get("U235_nu_fission", np.zeros_like(depth))
             - data.get("U235_fission", np.zeros_like(depth)))
            + (data.get("U238_nu_fission", np.zeros_like(depth))
               - data.get("U238_fission", np.zeros_like(depth)))
            + (data.get("Pu239_nu_fission", np.zeros_like(depth))
               - data.get("Pu239_fission", np.zeros_like(depth)))
            + (data.get("Am241_nu_fission", np.zeros_like(depth))
               - data.get("Am241_fission", np.zeros_like(depth)))
            + data.get("Am241_n2n", np.zeros_like(depth))
            + 2.0 * data.get("Am241_n3n", np.zeros_like(depth))
            + (oa_nuf - oa_fiss)
        ) - nXt - act_cap
        ax.plot(depth, total, color="black", linewidth=0.75, linestyle=":",
                label="Net cumulative balance")

        ax.axvspan(offset, offset + FUEL_SHELL_THICKNESS, color="gray",
                   alpha=0.15, label="Spent fuel shell")

        ax.axhline(0, color="black", linewidth=0.4)
        ax.set_xlabel("Radius [cm]")
        ax.set_ylabel("Cumulative neutrons gained / src-n")
        # ax.set_title(
        #     f"Cumulative neutron balance \u2014 "
        #     f"fuel shell midpoint = {midpt:.0f} cm"
        # )
        ax.legend(fontsize=2, loc="best", ncol=2)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, blanket_depth)

        leg = plt.legend(fancybox=False, edgecolor='black', frameon=True, framealpha=.75, ncol=2)
        leg.get_frame().set_linewidth(0.5) 

        plot_path = os.path.join(
            OUTPUT_DIR,
            f"uzr_zpinch_cumulative_nbal_mid{midpt:.0f}cm.pdf"
        )
        fig.savefig(plot_path, dpi=200)
        print(f"Cumulative neutron balance plot saved to: {plot_path}")
        plt.close(fig)


# ==============================================================================
# Main
# ==============================================================================

def main():
    blanket_depth = R_OUTER - R_INNER

    print("=" * 72)
    print("  Z-Pinch Cylindrical Wedge Model")
    print("  Zap Energy SFS Z-Pinch: Fuel Shell in PbLi Blanket")
    print("  Fuel zone: 30% spent U-10Zr / 20% HT-9 / 50% PbLi (by volume)")
    print("=" * 72)
    print(f"\n  Geometry:")
    print(f"    Inner radius (R_in):   {R_INNER} cm")
    print(f"    Outer radius (R_out):  {R_OUTER} cm")
    print(f"    Blanket depth:         {blanket_depth:.1f} cm")
    print(f"    Total height:          {2*HALF_HEIGHT:.0f} cm "
          f"(vacuum ceiling & floor)")
    print(f"    Wedge angle:           {WEDGE_ANGLE_DEG}° "
          f"(1/{360/WEDGE_ANGLE_DEG:.0f}th cylinder)")
    print(f"    Pinch length:          {2*PINCH_HALF_LENGTH:.0f} cm "
          f"(source extent)")
    print(f"\n  Fuel shell:")
    print(f"    Radial thickness:      {FUEL_SHELL_THICKNESS} cm")
    print(f"    Number of positions:   {len(SHELL_MIDPOINTS)}")
    print(f"    Midpoint sweep range:  "
          f"{SHELL_MIDPOINTS[0]:.1f} – {SHELL_MIDPOINTS[-1]:.1f} cm "
          f"from inner wall")
    print(f"\n  Monte Carlo:")
    print(f"    Particles per batch:   {PARTICLES}")
    print(f"    Batches:               {BATCHES}")
    print(f"    Source energy:          14.1 MeV D-T fusion neutrons")
    print(f"    Mesh tally resolution: {MESH_DR*10:.0f} mm "
          f"({N_MESH_R} radial bins)\n")

    # Build materials
    print("Building PbLi eutectic material (Li-17Pb-83)...")
    pbli_mat = build_pbli_material()

    print("Building HT-9 cladding material (Fe-12Cr-1MoVW)...")
    ht9_mat = build_ht9_material()

    print(f"Loading spent fuel inventory from '{CSV_PATH}'...")
    inventory = parse_inventory(CSV_PATH)
    print(f"  Loaded {len(inventory)} nuclides (noble gases excluded)")

    total = sum(inventory.values())
    sorted_nucs = sorted(inventory.items(), key=lambda x: x[1], reverse=True)
    print("\n  Top 10 nuclides by atom fraction:")
    for nuc, atoms in sorted_nucs[:10]:
        print(f"    {nuc:12s}  {atoms/total*100:8.4f}%")

    fuel_mat = build_spent_fuel_material(inventory)

    print(f"\n  Homogenizing fuel zone:")
    print(f"    Fuel slugs (U-10Zr):  {FUEL_VOL_FRAC*100:.0f} vol%  "
          f"@ {SPENT_FUEL_DENSITY} g/cm³")
    print(f"    HT-9 cladding:        {CLAD_VOL_FRAC*100:.0f} vol%  "
          f"@ {HT9_DENSITY} g/cm³")
    print(f"    PbLi coolant:         {COOLANT_VOL_FRAC*100:.0f} vol%  "
          f"@ {PBLI_DENSITY} g/cm³")
    fuel_zone_mat = build_fuel_zone_homogenized(fuel_mat, ht9_mat, pbli_mat)
    homog_density = (FUEL_VOL_FRAC * SPENT_FUEL_DENSITY
                     + CLAD_VOL_FRAC * HT9_DENSITY
                     + COOLANT_VOL_FRAC * PBLI_DENSITY)
    print(f"    Effective homog. density: {homog_density:.2f} g/cm³")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run parametric sweep
    all_results = []
    n_total = len(SHELL_MIDPOINTS) + 1

    # --- Baseline: pure PbLi, no spent fuel ---
    print(f"\n{chr(9472)*60}")
    print(f"  Case 1/{n_total}: NO FUEL (pure PbLi baseline)")
    print(f"{chr(9472)*60}")
    result = run_case(None, pbli_mat, fuel_zone_mat, 0)
    all_results.append(result)
    print(f"  TBR           = {result.get('TBR_mean', 0):.6f} "
          f"+/- {result.get('TBR_std', 0):.6f}")
    print(f"  (n,2n) rate   = {result.get('n2n_mean', 0):.6e} "
          f"+/- {result.get('n2n_std', 0):.6e}")
    print(f"  Leakage total = {result['leakage_mean']:.6e} "
          f"+/- {result['leakage_std']:.6e}")
    print(f"    outer wall  = {result['leakage_outer_mean']:.6e}")
    print(f"    ceiling     = {result['leakage_ceiling_mean']:.6e}")
    print(f"    floor       = {result['leakage_floor_mean']:.6e}")

    # --- Sweep fuel shell positions ---
    for i, midpt in enumerate(SHELL_MIDPOINTS):
        print(f"\n{chr(9472)*60}")
        print(f"  Case {i+2}/{n_total}: shell midpoint = {midpt:.1f} cm "
              f"from inner wall")
        print(f"{chr(9472)*60}")

        result = run_case(midpt, pbli_mat, fuel_zone_mat, i + 1)
        all_results.append(result)

        print(f"  TBR           = {result.get('TBR_mean', 0):.6f} "
              f"+/- {result.get('TBR_std', 0):.6f}")
        print(f"  Fission rate  = {result['fission_mean']:.6e} "
              f"\u00b1 {result['fission_std']:.6e}")
        print(f"  Capture rate  = {result['capture_mean']:.6e} "
              f"\u00b1 {result['capture_std']:.6e}")
        print(f"  (n,2n) rate   = {result.get('n2n_mean', 0):.6e} "
              f"\u00b1 {result.get('n2n_std', 0):.6e}")
        print(f"  Leakage total = {result['leakage_mean']:.6e} "
              f"\u00b1 {result['leakage_std']:.6e}")
        print(f"  k_inf         = {result.get('k_inf', 0):.6f} "
              f"\u00b1 {result.get('k_inf_std', 0):.6f}")
        if "mesh_csv" in result:
            print(f"  Mesh CSV      = {result['mesh_csv']}")

    # Save all results as JSON
    json_path = os.path.join(OUTPUT_DIR, "uzr_zpinch_all_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to: {json_path}")

    # Generate plots
    plot_results(all_results)
    plot_mesh_profiles(all_results)
    plot_neutron_balance(all_results)
    plot_cumulative_neutron_balance(all_results)

    # Print summary table
    print("\n" + "=" * 114)
    print(f"{'Midpoint [cm]':>14}  {'TBR':>12}  {'Fission':>12}  "
          f"{'Capture':>12}  {'(n,2n)':>12}  {'k_inf':>12}  {'Leak tot':>12}  "
          f"{'Leak axial':>12}")
    print("-" * 114)
    for r in all_results:
        mid = r['shell_midpoint_cm']
        label = f"{mid:14.1f}" if mid is not None else "    (no fuel) "
        axial_leak = (r.get('leakage_ceiling_mean', 0)
                      + r.get('leakage_floor_mean', 0))
        print(f"{label}  "
              f"{r.get('TBR_mean',0):12.6f}  "
              f"{r['fission_mean']:12.6e}  "
              f"{r['capture_mean']:12.6e}  "
              f"{r.get('n2n_mean',0):12.6e}  "
              f"{r.get('k_inf',0):12.6f}  "
              f"{r['leakage_mean']:12.6e}  "
              f"{axial_leak:12.6e}")
    print("=" * 114)
    print("\nDone.")


if __name__ == "__main__":
    main()