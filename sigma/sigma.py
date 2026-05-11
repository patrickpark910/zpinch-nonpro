#!/usr/bin/env python3
"""
Sigma Pile Sensitivity Analysis: Spent Fuel Slab Position in Zap Energy Z-Pinch Blanket
========================================================================================

This script models a 1x1 cm rectangular prism of total length 150 cm (the approximate
blanket/shield radius of Zap Energy's SFS Z-pinch fusion reactor, based on the 3 m
diameter shielding vessel from the Fusion Core Concept design). The prism is filled
with eutectic PbLi (Li-17Pb-83), and a 1 cm slab of spent fuel (loaded from the
provided CSV inventory) is inserted at varying distances from the 14.1 MeV neutron
source face. The script sweeps the slab position across the blanket depth and tracks
key nuclear tallies at each position.

Geometry (for a given slab offset d):
  |<-- d -->|<-1cm->|<---- 150 - d - 1 ---->|
  [  PbLi   ][ Fuel ][        PbLi           ]
  ^                                           ^
  source face                            back face

Tallies tracked:
  1. Tritium breeding rate in PbLi (Li6(n,t) + Li7(n,nt))
  2. Fission rate in spent fuel slab
  3. (n,gamma) capture rate in spent fuel
  4. (n,2n) reaction rate in PbLi (Pb neutron multiplication)
  5. Neutron flux spectrum in spent fuel slab (725-group)
  6. Heating (energy deposition) in each region
  7. Neutron leakage (current) through the back face

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
# ==============================================================================
# Configuration
# ==============================================================================
CSV_PATH = "bol_inventory_zpinch.csv"         # Path to spent fuel inventory CSV
TOTAL_LENGTH = 150.0                          # Total prism length [cm] — blanket radius
CROSS_SECTION = 1.0                           # Cross section width/height [cm]
FUEL_SLAB_THICKNESS = 1.0                     # Spent fuel slab thickness [cm]
# Slab midpoint positions [cm] — distance from the source face to the
# centre of the 1 cm spent fuel slab.  Sweep covers the middle third of
# the blanket so there is always PbLi on both sides of the slab.
SLAB_MIDPOINTS = np.arange(
    TOTAL_LENGTH / 5.0,                          # ~30 cm
    4.0 * TOTAL_LENGTH / 5.0 + 1.0,             # ~100 cm (inclusive)
    5.0,
)
MESH_DZ = 0.1                                 # Mesh tally z-resolution [cm] (1 mm)
N_MESH_Z = int(TOTAL_LENGTH / MESH_DZ)       # Number of mesh bins along z (1500)
BATCHES = 50                                  # Number of batches
INACTIVE_BATCHES = 10                         # Inactive batches (for fixed-source, set to 0)
PARTICLES = 1000                             # Particles per batch
# PbLi eutectic parameters (Li-17Pb-83 by atom fraction)
PBLI_LI_ATOM_FRAC = 0.17                     # 17 at% Li (natural: 7.5% Li-6, 92.5% Li-7)
PBLI_PB_ATOM_FRAC = 0.83                     # 83 at% Pb (natural isotopic mix)
PBLI_DENSITY = 9.49                           # g/cm³ at ~500°C operating temperature
# Spent fuel density — U-10Zr metallic alloy (consistent with ZFFR blanket design)
SPENT_FUEL_DENSITY = 15.8                     # g/cm³
OUTPUT_DIR = "sigma_pile_results"
# Nuclide groups for per-nuclide mesh tallies (neutron balance breakdown)
KEY_ACTINIDES = ["U235", "U238", "Pu239"]
OTHER_ACTINIDES = [
    "Am241", "Am243", "Np237", "Pu238", "Pu240", "Pu241", "Pu242",
    "U234", "U236",
]
ALL_ACTINIDES = KEY_ACTINIDES + OTHER_ACTINIDES
PB_ISOTOPES = ["Pb204", "Pb206", "Pb207", "Pb208"]


# ==============================================================================
# Parse spent fuel inventory from CSV
# ==============================================================================

def parse_inventory(csv_path):
    """
    Read the BOL inventory CSV and return a dict of {openmc_nuclide_name: atom_count}.
    
    Metastable states (e.g. 'Am242_m1', 'Nb93_m1') are folded into their
    ground-state counterparts because most metastable isomers do not have
    separate cross-section evaluations in standard ENDF libraries.  The few
    that do (Am242_m1) are present at negligible atom fractions relative to
    the ground state and the merge introduces no meaningful error.
    
    Noble gases (He, Kr, Xe) are skipped — they escape metallic fuel.
    """
    inventory = {}
    noble_gas_elements = {"He", "Kr", "Xe"}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_name = row["nuclide"].strip()
            atoms = float(row["atoms"].strip())

            # Extract element symbol
            match = re.match(r"([A-Z][a-z]?)", raw_name)
            if not match:
                continue
            element = match.group(1)

            # Skip noble gases (volatile fission products that escape metallic fuel)
            if element in noble_gas_elements:
                continue

            # Fold metastable states into ground state:
            #   'Am242_m1' -> 'Am242',  'Nb93_m1' -> 'Nb93', etc.
            nuclide_name = re.sub(r"_m\d+$", "", raw_name)

            # Accumulate (ground + metastable atoms go into the same bucket)
            inventory[nuclide_name] = inventory.get(nuclide_name, 0.0) + atoms

    return inventory


def build_spent_fuel_material(inventory):
    """
    Build an OpenMC Material from the atom inventory.
    Uses atom fractions derived from the total atom counts.
    """
    total_atoms = sum(inventory.values())
    fuel = openmc.Material(name="Spent_Fuel_U10Zr")
    fuel.set_density("g/cm3", SPENT_FUEL_DENSITY)

    for nuclide, atoms in inventory.items():
        atom_frac = atoms / total_atoms
        # Only include nuclides with meaningful contribution
        if atom_frac < 1.0e-12:
            continue
        try:
            fuel.add_nuclide(nuclide, atom_frac, percent_type="ao")
        except Exception:
            # If a nuclide isn't in the cross-section library, skip it
            print(f"  Warning: Nuclide '{nuclide}' not found in library, skipping.")

    return fuel


# ==============================================================================
# Build PbLi eutectic material
# ==============================================================================

def build_pbli_material():
    """
    Build eutectic PbLi (Li-17Pb-83 by atom %) using natural isotopic abundances.
    """
    pbli = openmc.Material(name="PbLi_eutectic")
    pbli.set_density("g/cm3", PBLI_DENSITY)

    # Lithium (natural: 7.59% Li6, 92.41% Li7)
    pbli.add_nuclide("Li6",  PBLI_LI_ATOM_FRAC * 0.0759,  "ao")
    pbli.add_nuclide("Li7",  PBLI_LI_ATOM_FRAC * 0.9241,  "ao")

    # Lead (natural isotopic abundances)
    pbli.add_nuclide("Pb204", PBLI_PB_ATOM_FRAC * 0.014,  "ao")
    pbli.add_nuclide("Pb206", PBLI_PB_ATOM_FRAC * 0.241,  "ao")
    pbli.add_nuclide("Pb207", PBLI_PB_ATOM_FRAC * 0.221,  "ao")
    pbli.add_nuclide("Pb208", PBLI_PB_ATOM_FRAC * 0.524,  "ao")

    return pbli


# ==============================================================================
# Mesh tally I/O helpers
# ==============================================================================

def _save_mesh_csv(path, mesh_data):
    """
    Write the depth-resolved mesh tally data to a CSV file.

    Parameters
    ----------
    path : str
        Output file path.
    mesh_data : dict
        Keys are column names; values are 1-D numpy arrays (all same length)
        or a scalar 'z_cm' array for the bin centres.
    """
    z = mesh_data["z_cm"]
    n = len(z)
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


# ==============================================================================
# Build geometry and run for a given slab position
# ==============================================================================

def run_case(slab_midpoint, pbli_mat, fuel_mat, run_index):
    """
    Build and run an OpenMC model for the spent fuel slab centred at a
    given depth in the blanket.  Pass *slab_midpoint=None* for a pure-PbLi
    baseline (no spent fuel).

    Parameters
    ----------
    slab_midpoint : float or None
        Distance [cm] from the source face to the *midpoint* of the
        1 cm spent fuel slab, or None for pure-PbLi reference case.
    pbli_mat : openmc.Material
        The PbLi blanket material.
    fuel_mat : openmc.Material
        The spent fuel material (ignored when slab_midpoint is None).
    run_index : int
        Index for the parametric sweep (used for directory naming).

    Returns
    -------
    dict : Tally results for this case.
    """

    no_fuel = slab_midpoint is None

    if no_fuel:
        slab_offset = None
        case_dir = os.path.join(OUTPUT_DIR, f"case_{run_index:03d}_no_fuel")
    else:
        slab_offset = slab_midpoint - FUEL_SLAB_THICKNESS / 2.0
        case_dir = os.path.join(OUTPUT_DIR, f"case_{run_index:03d}_mid_{slab_midpoint:.1f}cm")
    os.makedirs(case_dir, exist_ok=True)

    # Reset OpenMC's global auto-ID counters so that Materials, Surfaces,
    # Cells, Filters, and Tallies start from id=1 in every case.  Without
    # this, IDs accumulate across the parametric loop and trigger IDWarnings.
    try:
        openmc.reset_auto_ids()
    except AttributeError:
        # Older OpenMC versions lack reset_auto_ids; warnings are cosmetic only.
        pass

    # ------------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------------
    # Clone materials so each case is independent
    pbli_front = pbli_mat.clone()
    pbli_front.name = "PbLi_front"

    if no_fuel:
        materials = openmc.Materials([pbli_front])
    else:
        pbli_back = pbli_mat.clone()
        pbli_back.name = "PbLi_back"
        fuel = fuel_mat.clone()
        fuel.name = "Spent_Fuel"
        materials = openmc.Materials([pbli_front, pbli_back, fuel])

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    half_w = CROSS_SECTION / 2.0

    # Surfaces — axial planes along the z-axis (beam direction)
    # The source face is reflective: in the real Z-pinch geometry, neutrons
    # heading back toward the plasma pass through and hit PbLi on the far
    # side, so reflection approximates the symmetric radial coverage.
    z_source     = openmc.ZPlane(z0=0.0,                        boundary_type="reflective")
    z_end        = openmc.ZPlane(z0=TOTAL_LENGTH,               boundary_type="vacuum")

    # Lateral surfaces (reflective to approximate infinite slab)
    x_min = openmc.XPlane(x0=-half_w, boundary_type="reflective")
    x_max = openmc.XPlane(x0=+half_w, boundary_type="reflective")
    y_min = openmc.YPlane(y0=-half_w, boundary_type="reflective")
    y_max = openmc.YPlane(y0=+half_w, boundary_type="reflective")

    lateral = +x_min & -x_max & +y_min & -y_max

    cells = []

    if no_fuel:
        # Pure PbLi — single cell fills the entire prism
        region_all = +z_source & -z_end & lateral
        cell_all = openmc.Cell(name="PbLi_full", fill=pbli_front, region=region_all)
        cells.append(cell_all)
    else:
        z_fuel_front = openmc.ZPlane(z0=slab_offset)
        z_fuel_back  = openmc.ZPlane(z0=slab_offset + FUEL_SLAB_THICKNESS)

        has_front = slab_offset > 0.0
        has_back  = (slab_offset + FUEL_SLAB_THICKNESS) < TOTAL_LENGTH

        # Front PbLi region
        if has_front:
            region_front = +z_source & -z_fuel_front & lateral
            cell_front = openmc.Cell(name="PbLi_front", fill=pbli_front, region=region_front)
            cells.append(cell_front)

        # Spent fuel slab
        region_fuel = +z_fuel_front & -z_fuel_back & lateral
        cell_fuel = openmc.Cell(name="Spent_Fuel_Slab", fill=fuel, region=region_fuel)
        cells.append(cell_fuel)

        # Back PbLi region
        if has_back:
            region_back = +z_fuel_back & -z_end & lateral
            cell_back = openmc.Cell(name="PbLi_back", fill=pbli_back, region=region_back)
            cells.append(cell_back)

    universe = openmc.Universe(cells=cells)
    geometry = openmc.Geometry(universe)

    # ------------------------------------------------------------------
    # Source: 14.1 MeV monoenergetic neutrons, entering from z=0 face
    # ------------------------------------------------------------------
    source = openmc.IndependentSource()
    source.space = openmc.stats.Box(
        lower_left=(-half_w, -half_w, 0.0),
        upper_right=(+half_w, +half_w, 0.0),
        only_fissionable=False
    )
    source.angle = openmc.stats.Monodirectional(reference_uvw=(0.0, 0.0, 1.0))
    source.energy = openmc.stats.Discrete([14.1e6], [1.0])

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.batches = BATCHES
    settings.inactive = 0  # No inactive batches for fixed source
    settings.particles = PARTICLES
    settings.source = source

    # ------------------------------------------------------------------
    # Tallies
    # ------------------------------------------------------------------
    tallies = openmc.Tallies()

    # --- Tally 1: Tritium Breeding Rate in ALL PbLi ---
    # Li6(n,t) = MT 205, Li7(n,nt) = MT 205 for Li7... use (n,Xt) score
    if no_fuel:
        pbli_filter_mats = [pbli_front]
    else:
        pbli_filter_mats = []
        if has_front:
            pbli_filter_mats.append(pbli_front)
        if has_back:
            pbli_filter_mats.append(pbli_back)

    if pbli_filter_mats:
        tbr_tally = openmc.Tally(name="TBR")
        tbr_tally.filters = [openmc.MaterialFilter(pbli_filter_mats)]
        tbr_tally.scores = ["(n,Xt)"]
        tallies.append(tbr_tally)

    # --- Tally 2: Fission rate in spent fuel (skip if no fuel) ---
    if not no_fuel:
        fission_tally = openmc.Tally(name="fission_rate")
        fission_tally.filters = [openmc.MaterialFilter([fuel])]
        fission_tally.scores = ["fission"]
        tallies.append(fission_tally)

    # --- Tally 3: Radiative capture in spent fuel (skip if no fuel) ---
    if not no_fuel:
        capture_tally = openmc.Tally(name="capture_rate")
        capture_tally.filters = [openmc.MaterialFilter([fuel])]
        capture_tally.scores = ["(n,gamma)"]
        tallies.append(capture_tally)

    # --- Tally 4: (n,2n) in PbLi (Pb neutron multiplication) ---
    if pbli_filter_mats:
        n2n_tally = openmc.Tally(name="n2n_rate")
        n2n_tally.filters = [openmc.MaterialFilter(pbli_filter_mats)]
        n2n_tally.scores = ["(n,2n)"]
        tallies.append(n2n_tally)

    # --- Tally 5: Neutron flux spectrum in spent fuel (skip if no fuel) ---
    energy_bins = np.logspace(np.log10(1e-5), np.log10(20e6), 726)  # 725 groups
    if not no_fuel:
        energy_filter = openmc.EnergyFilter(energy_bins)
        spectrum_tally = openmc.Tally(name="fuel_spectrum")
        spectrum_tally.filters = [openmc.MaterialFilter([fuel]), energy_filter]
        spectrum_tally.scores = ["flux"]
        tallies.append(spectrum_tally)

    # --- Tally 6: Heating in each region ---
    all_mats = pbli_filter_mats + ([fuel] if not no_fuel else [])
    heating_tally = openmc.Tally(name="heating")
    heating_tally.filters = [openmc.MaterialFilter(all_mats)]
    heating_tally.scores = ["heating"]  # eV/source-particle
    tallies.append(heating_tally)

    # --- Tally 7: Neutron current (leakage) through back face ---
    back_surface_filter = openmc.SurfaceFilter([z_end])
    leakage_tally = openmc.Tally(name="leakage")
    leakage_tally.filters = [back_surface_filter]
    leakage_tally.scores = ["current"]
    tallies.append(leakage_tally)

    # --- Tally 8: Absorption in spent fuel (skip if no fuel) ---
    if not no_fuel:
        abs_tally = openmc.Tally(name="absorption_fuel")
        abs_tally.filters = [openmc.MaterialFilter([fuel])]
        abs_tally.scores = ["absorption"]
        tallies.append(abs_tally)

    # ------------------------------------------------------------------
    # Mesh Tally: 1 mm depth-resolved profiles across the full blanket
    # ------------------------------------------------------------------
    # A single mesh bin in x and y (integrated over the 1×1 cm cross section),
    # with N_MESH_Z bins along z at MESH_DZ resolution (default 1 mm).
    # This gives a depth profile of every reaction rate through the blanket.
    mesh = openmc.RegularMesh()
    mesh.dimension = [1, 1, N_MESH_Z]
    mesh.lower_left = [-half_w, -half_w, 0.0]
    mesh.upper_right = [+half_w, +half_w, TOTAL_LENGTH]
    mesh_filter = openmc.MeshFilter(mesh)

    # All 8 scores on one mesh tally (volume-compatible scores).
    # Note: "current" is a surface-only score, so for the depth-resolved
    # leakage equivalent we use "flux" — the attenuation profile provides
    # the same physical insight. We add "nu-fission" as a bonus to track
    # the fission neutron source distribution.
    MESH_SCORES = [
        "(n,Xt)",       # 1. Tritium breeding rate
        "fission",      # 2. Fission rate
        "(n,gamma)",    # 3. Radiative capture
        "(n,2n)",       # 4. Neutron multiplication (Pb)
        "flux",         # 5. Neutron flux (depth profile)
        "heating",      # 6. Energy deposition [eV/src]
        "absorption",   # 7. Total absorption
        "nu-fission",   # 8. Fission neutron production (ν·Σf·φ)
    ]

    mesh_tally = openmc.Tally(name="mesh_depth_profile")
    mesh_tally.filters = [mesh_filter]
    mesh_tally.scores = MESH_SCORES
    tallies.append(mesh_tally)

    # Also add a mesh tally with a surface current score on the mesh
    # to capture the net z-directed neutron current at each depth plane.
    mesh_surface_filter = openmc.MeshSurfaceFilter(mesh)
    current_mesh_tally = openmc.Tally(name="mesh_current_profile")
    current_mesh_tally.filters = [mesh_surface_filter]
    current_mesh_tally.scores = ["current"]
    tallies.append(current_mesh_tally)

    # ------------------------------------------------------------------
    # Per-nuclide mesh tallies for neutron balance breakdown
    # ------------------------------------------------------------------
    if not no_fuel:
        # Tally A: key actinides — fission, nu-fission, capture, (n,2n)
        nuc_key_tally = openmc.Tally(name="mesh_key_actinides")
        nuc_key_tally.filters = [mesh_filter]
        nuc_key_tally.nuclides = KEY_ACTINIDES
        nuc_key_tally.scores = ["fission", "nu-fission", "(n,gamma)", "(n,2n)"]
        tallies.append(nuc_key_tally)

        # Tally B: other actinides — fission, nu-fission, capture
        nuc_other_tally = openmc.Tally(name="mesh_other_actinides")
        nuc_other_tally.filters = [mesh_filter]
        nuc_other_tally.nuclides = OTHER_ACTINIDES
        nuc_other_tally.scores = ["fission", "nu-fission", "(n,gamma)"]
        tallies.append(nuc_other_tally)

    # Tally C: Pb (n,2n) — always present (PbLi is in every case)
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
            print(f"  Statepoint '{sp_filename}' already exists — skipping OpenMC run.")
        else:
            model.export_to_model_xml()
            openmc.run(output=True)

        # Extract results
        sp_file = openmc.StatePoint(sp_filename)
        results = {
            "slab_midpoint_cm": slab_midpoint,
            "slab_offset_cm": slab_offset,
            "no_fuel": no_fuel,
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
        t = sp_file.get_tally(name="heating")
        results["heating_eV_per_sp"] = {
            m.name: float(t.mean[i][0][0])
            for i, m in enumerate(all_mats)
        }

        # Leakage
        t = sp_file.get_tally(name="leakage")
        results["leakage_mean"] = float(t.mean.sum())
        results["leakage_std"]  = float(np.sqrt((t.std_dev**2).sum()))

        # Absorption in fuel
        if not no_fuel:
            t = sp_file.get_tally(name="absorption_fuel")
            results["absorption_mean"] = float(t.mean.sum())
            results["absorption_std"]  = float(np.sqrt((t.std_dev**2).sum()))
        else:
            results["absorption_mean"] = results["absorption_std"] = 0.0

        # -----------------------------------------------------------
        # Extract depth-resolved mesh tallies
        # -----------------------------------------------------------
        z_edges = np.linspace(0.0, TOTAL_LENGTH, N_MESH_Z + 1)
        z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

        # Volume mesh tally (8 scores)
        mt = sp_file.get_tally(name="mesh_depth_profile")
        # OpenMC stores tally.mean as a flat (n_total_bins, 1, 1) array
        # where n_total_bins = N_MESH_Z * n_scores.  Reshape to separate
        # mesh bins (rows) from score bins (columns).
        n_scores = len(MESH_SCORES)
        mt_mean = mt.mean.flatten().reshape(N_MESH_Z, n_scores)
        mt_std  = mt.std_dev.flatten().reshape(N_MESH_Z, n_scores)

        mesh_data = {"z_cm": z_centers}
        for si, score_name in enumerate(MESH_SCORES):
            # Sanitise score name for use as a column header
            col = score_name.replace("(", "").replace(")", "").replace(",", "").replace("-", "_")
            mesh_data[f"{col}_mean"] = mt_mean[:, si]
            mesh_data[f"{col}_std"]  = mt_std[:, si]

        # Surface-current mesh tally (net z-current at each depth plane)
        ct = sp_file.get_tally(name="mesh_current_profile")
        ct_mean = ct.mean.flatten()
        ct_std  = ct.std_dev.flatten()
        # MeshSurfaceFilter produces (N_MESH_Z * 2*ndim) surface bins for
        # a 3-D mesh: 2 surfaces per dimension per voxel.  For a [1,1,Nz]
        # mesh the z-normal surfaces are the ones we want.  In OpenMC the
        # surface ordering per voxel is (-x, +x, -y, +y, -z, +z), so the
        # z-outward (+z) surface of voxel i is at index 6*i + 5.
        n_surf_per_vox = 6  # 2 surfaces × 3 dimensions
        z_plus_indices = np.arange(N_MESH_Z) * n_surf_per_vox + 5
        # Guard against shape mismatches from different OpenMC versions
        if z_plus_indices[-1] < len(ct_mean):
            mesh_data["current_z_mean"] = ct_mean[z_plus_indices]
            mesh_data["current_z_std"]  = ct_std[z_plus_indices]
        else:
            # Fallback: store raw current array for manual inspection
            np.save("mesh_current_raw_mean.npy", ct_mean)
            np.save("mesh_current_raw_std.npy",  ct_std)
            print("  Note: MeshSurface current saved as raw .npy for manual slicing.")

        # Save mesh data as CSV
        mesh_csv_path = "mesh_depth_profiles.csv"
        _save_mesh_csv(mesh_csv_path, mesh_data)

        # ----------------------------------------------------------
        # Per-nuclide mesh tallies for neutron balance
        # ----------------------------------------------------------
        nbal = {"z_cm": z_centers}

        # Pb (n,2n) — always available
        pb_t = sp_file.get_tally(name="mesh_pb_n2n")
        pb_data = pb_t.mean.flatten().reshape(N_MESH_Z, len(PB_ISOTOPES), 1)
        nbal["pb_n2n"] = pb_data[:, :, 0].sum(axis=1)  # sum over Pb isotopes

        if not no_fuel:
            # Key actinides: shape (N_MESH_Z, 3 nuclides, 4 scores)
            ka_t = sp_file.get_tally(name="mesh_key_actinides")
            ka_scores = ["fission", "nu-fission", "(n,gamma)", "(n,2n)"]
            ka = ka_t.mean.flatten().reshape(N_MESH_Z, len(KEY_ACTINIDES), len(ka_scores))
            for ni, nuc in enumerate(KEY_ACTINIDES):
                nbal[f"{nuc}_fission"]    = ka[:, ni, 0]
                nbal[f"{nuc}_nu_fission"] = ka[:, ni, 1]
                nbal[f"{nuc}_ngamma"]     = ka[:, ni, 2]
                nbal[f"{nuc}_n2n"]        = ka[:, ni, 3]

            # Other actinides: shape (N_MESH_Z, N_other, 3 scores)
            oa_t = sp_file.get_tally(name="mesh_other_actinides")
            oa_scores = ["fission", "nu-fission", "(n,gamma)"]
            oa = oa_t.mean.flatten().reshape(N_MESH_Z, len(OTHER_ACTINIDES), len(oa_scores))
            # Sum over all "other" actinides
            nbal["other_act_fission"]    = oa[:, :, 0].sum(axis=1)
            nbal["other_act_nu_fission"] = oa[:, :, 1].sum(axis=1)
            nbal["other_act_ngamma"]     = oa[:, :, 2].sum(axis=1)

        # (n,Xt) from the main mesh tally — already in mesh_data
        nbal["nXt"] = mt_mean[:, 0]  # score index 0 = (n,Xt)

        # Save neutron balance CSV
        nbal_csv_path = "neutron_balance_depth.csv"
        _save_mesh_csv(nbal_csv_path, nbal)
        results["nbal_csv"] = os.path.join(case_dir, nbal_csv_path)

        # Attach summary arrays to results dict (not the full 1500-bin arrays,
        # just the path so downstream code can find them)
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
        print("matplotlib not available — skipping plots. Results saved as JSON.")
        return

    # Filter to fuel-only cases (skip the no-fuel baseline for these plots)
    fuel_only = [r for r in all_results if not r.get("no_fuel", False)]
    if not fuel_only:
        return

    offsets      = [r["slab_midpoint_cm"] for r in fuel_only]
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
        "Sigma Pile Sensitivity: Spent Fuel Slab Position in Zap Energy Z-Pinch PbLi Blanket",
        fontsize=14, fontweight="bold"
    )

    # (a) Tritium Breeding Rate
    ax = axes[0, 0]
    ax.errorbar(offsets, tbr_mean, yerr=tbr_std, fmt="o-", capsize=3, color="tab:blue")
    ax.set_xlabel("Spent fuel slab midpoint depth [cm]")
    ax.set_ylabel("Tritium Production Rate\n[per source neutron]")
    ax.set_title("(a) Tritium Breeding Rate in PbLi")
    ax.grid(True, alpha=0.3)

    # (b) Fission Rate in Fuel
    ax = axes[0, 1]
    ax.errorbar(offsets, fiss_mean, yerr=fiss_std, fmt="s-", capsize=3, color="tab:red")
    ax.set_xlabel("Spent fuel slab midpoint depth [cm]")
    ax.set_ylabel("Fission Rate\n[per source neutron]")
    ax.set_title("(b) Fission Rate in Spent Fuel")
    ax.grid(True, alpha=0.3)

    # (c) Capture Rate in Fuel
    ax = axes[1, 0]
    ax.errorbar(offsets, cap_mean, yerr=cap_std, fmt="^-", capsize=3, color="tab:green")
    ax.set_xlabel("Spent fuel slab midpoint depth [cm]")
    ax.set_ylabel("(n,γ) Capture Rate\n[per source neutron]")
    ax.set_title("(c) Radiative Capture in Spent Fuel")
    ax.grid(True, alpha=0.3)

    # (d) (n,2n) Rate in PbLi
    ax = axes[1, 1]
    ax.errorbar(offsets, n2n_mean, yerr=n2n_std, fmt="D-", capsize=3, color="tab:orange")
    ax.set_xlabel("Spent fuel slab midpoint depth [cm]")
    ax.set_ylabel("(n,2n) Rate in PbLi\n[per source neutron]")
    ax.set_title("(d) Pb Neutron Multiplication")
    ax.grid(True, alpha=0.3)

    # (e) Neutron Leakage Through Back Face
    ax = axes[2, 0]
    ax.errorbar(offsets, leak_mean, yerr=leak_std, fmt="v-", capsize=3, color="tab:purple")
    ax.set_xlabel("Spent fuel slab midpoint depth [cm]")
    ax.set_ylabel("Neutron Current (Leakage)\n[per source neutron]")
    ax.set_title("(e) Leakage Through Back Face")
    ax.grid(True, alpha=0.3)

    # (f) Total Absorption in Fuel
    ax = axes[2, 1]
    ax.errorbar(offsets, abs_mean, yerr=abs_std, fmt="p-", capsize=3, color="tab:brown")
    ax.set_xlabel("Spent fuel slab midpoint depth [cm]")
    ax.set_ylabel("Total Absorption in Fuel\n[per source neutron]")
    ax.set_title("(f) Total Neutron Absorption in Spent Fuel")
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(OUTPUT_DIR, "sigma_pile_sensitivity.png")
    fig.savefig(plot_path, dpi=200)
    print(f"\nPlots saved to: {plot_path}")
    plt.close(fig)

    # --- Additional plot: select neutron spectra at a few slab positions ---
    fuel_only = [r for r in all_results if not r.get("no_fuel", False)]
    if fuel_only:
        fig2, ax2 = plt.subplots(figsize=(10, 6), constrained_layout=True)
        indices_to_plot = np.linspace(0, len(fuel_only) - 1, min(6, len(fuel_only)), dtype=int)
        cmap = plt.cm.viridis(np.linspace(0, 1, len(indices_to_plot)))

        for color_idx, i in enumerate(indices_to_plot):
            r = fuel_only[i]
            e_bins = np.array(r["energy_bins_eV"])
            e_mid = 0.5 * (e_bins[:-1] + e_bins[1:])
            lethargy_width = np.log(e_bins[1:] / e_bins[:-1])
            flux = np.array(r["fuel_spectrum_mean"])
            flux_per_lethargy = flux / lethargy_width
            ax2.loglog(e_mid, flux_per_lethargy, color=cmap[color_idx],
                       label=f"d = {r['slab_midpoint_cm']:.0f} cm", alpha=0.8)

        ax2.set_xlabel("Neutron Energy [eV]")
        ax2.set_ylabel("Flux per unit lethargy [n/src/lethargy]")
        ax2.set_title("Neutron Spectrum in Spent Fuel Slab vs. Position")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, which="both")

        spec_path = os.path.join(OUTPUT_DIR, "fuel_spectra_vs_position.png")
        fig2.savefig(spec_path, dpi=200)
        print(f"Spectra plot saved to: {spec_path}")
        plt.close(fig2)


def plot_mesh_profiles(all_results):
    """
    Generate depth-resolved mesh tally plots for a selection of slab positions.
    Reads the per-case mesh CSV files and overlays profiles on shared axes.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping mesh plots.")
        return

    # Pick a subset of cases to overlay (avoid clutter) — skip no-fuel baseline
    fuel_results = [r for r in all_results if not r.get("no_fuel", False)]
    n_cases = len(fuel_results)
    if n_cases == 0:
        return
    indices = np.linspace(0, n_cases - 1, min(6, n_cases), dtype=int)
    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(indices)))

    # Score columns to plot and their display labels / y-axis labels
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
        "1 mm Depth-Resolved Mesh Tallies — Zap Z-Pinch PbLi Blanket",
        fontsize=14, fontweight="bold",
    )

    for panel_idx, (col, title, ylabel) in enumerate(score_panels):
        ax = axes.flat[panel_idx]
        for ci, case_i in enumerate(indices):
            r = fuel_results[case_i]
            csv_path = r.get("mesh_csv")
            if csv_path is None or not os.path.exists(csv_path):
                continue

            # Read the mesh CSV
            data = _read_mesh_csv(csv_path)
            z = data.get("z_cm")
            y = data.get(col)
            if z is None or y is None:
                continue

            offset = r["slab_offset_cm"]
            midpt  = r["slab_midpoint_cm"]
            ax.semilogy(z, y, color=cmap[ci], alpha=0.75, linewidth=0.8,
                        label=f"d = {midpt:.0f} cm")

            # Shade the spent fuel slab location
            ax.axvspan(offset, offset + FUEL_SLAB_THICKNESS,
                       color=cmap[ci], alpha=0.08)

        ax.set_xlabel("Depth into blanket [cm]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best", ncol=2)
        ax.grid(True, alpha=0.25, which="both")
        ax.set_xlim(0, TOTAL_LENGTH)

    mesh_plot_path = os.path.join(OUTPUT_DIR, "mesh_depth_profiles.png")
    fig.savefig(mesh_plot_path, dpi=200)
    print(f"Mesh depth profile plots saved to: {mesh_plot_path}")
    plt.close(fig)

    # --- Also plot the z-current profile if available ---
    fig3, ax3 = plt.subplots(figsize=(10, 5), constrained_layout=True)
    found_current = False
    for ci, case_i in enumerate(indices):
        r = fuel_results[case_i]
        csv_path = r.get("mesh_csv")
        if csv_path is None or not os.path.exists(csv_path):
            continue
        data = _read_mesh_csv(csv_path)
        z = data.get("z_cm")
        j = data.get("current_z_mean")
        if z is None or j is None:
            continue
        found_current = True
        offset = r["slab_offset_cm"]
        midpt  = r["slab_midpoint_cm"]
        ax3.semilogy(z, j, color=cmap[ci], alpha=0.75, linewidth=0.8,
                     label=f"d = {midpt:.0f} cm")
        ax3.axvspan(offset, offset + FUEL_SLAB_THICKNESS,
                    color=cmap[ci], alpha=0.08)

    if found_current:
        ax3.set_xlabel("Depth into blanket [cm]")
        ax3.set_ylabel("Net z-directed current [per src / surface]")
        ax3.set_title("Neutron Current (z-direction) Depth Profile")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.25, which="both")
        ax3.set_xlim(0, TOTAL_LENGTH)
        current_path = os.path.join(OUTPUT_DIR, "mesh_z_current_profile.png")
        fig3.savefig(current_path, dpi=200)
        print(f"Z-current profile plot saved to: {current_path}")
    plt.close(fig3)


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



def plot_neutron_balance(all_results):
    """
    Plot the depth-resolved neutron balance: gains and losses per 1 mm bin.

    Gains (positive):
      - Pb (n,2n):  +1 neutron per reaction
      - U235 fission net:  nu-fission(U235) - fission(U235)
      - U238 fission net:  nu-fission(U238) - fission(U238)
      - Pu239 fission net: nu-fission(Pu239) - fission(Pu239)
      - Other actinide fission net

    Losses (negative):
      - (n,Xt) tritium breeding: -1 per reaction
      - Actinide (n,gamma) capture (all actinides summed)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- skipping neutron balance plot.")
        return

    fuel_results = [r for r in all_results if not r.get("no_fuel", False)]
    if not fuel_results:
        return

    # Pick a subset of cases to plot
    n_cases = len(fuel_results)
    indices = np.linspace(0, n_cases - 1, min(4, n_cases), dtype=int)

    for ci in indices:
        r = fuel_results[ci]
        csv_path = r.get("nbal_csv")
        if csv_path is None or not os.path.exists(csv_path):
            continue

        data = _read_mesh_csv(csv_path)
        z = data.get("z_cm")
        if z is None:
            continue

        midpt = r["slab_midpoint_cm"]
        offset = r["slab_offset_cm"]

        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

        # --- GAINS (positive) ---
        pb_n2n = data.get("pb_n2n", np.zeros_like(z))
        ax.fill_between(z, 0, pb_n2n, alpha=0.3, color="tab:blue", label="Pb (n,2n) gain")
        ax.plot(z, pb_n2n, color="tab:blue", linewidth=0.6)

        for nuc, color in [("U235", "tab:red"), ("U238", "tab:orange"), ("Pu239", "tab:green")]:
            fiss = data.get(f"{nuc}_fission", np.zeros_like(z))
            nu_f = data.get(f"{nuc}_nu_fission", np.zeros_like(z))
            net = nu_f - fiss  # net neutrons gained from fission
            ax.plot(z, net, color=color, linewidth=1.2, label=f"{nuc} fission net gain")

        oa_fiss = data.get("other_act_fission", np.zeros_like(z))
        oa_nuf  = data.get("other_act_nu_fission", np.zeros_like(z))
        oa_net  = oa_nuf - oa_fiss
        if oa_net.max() > 1e-15:
            ax.plot(z, oa_net, color="tab:olive", linewidth=1.0, linestyle="--",
                    label="Other actinide fission net gain")

        # --- LOSSES (negative) ---
        nXt = data.get("nXt", np.zeros_like(z))
        ax.fill_between(z, 0, -nXt, alpha=0.2, color="tab:purple", label="(n,Xt) tritium loss")
        ax.plot(z, -nXt, color="tab:purple", linewidth=0.6)

        # Sum all actinide (n,gamma)
        act_cap = np.zeros_like(z)
        for nuc in KEY_ACTINIDES:
            act_cap += data.get(f"{nuc}_ngamma", np.zeros_like(z))
        act_cap += data.get("other_act_ngamma", np.zeros_like(z))
        ax.plot(z, -act_cap, color="tab:brown", linewidth=1.2, label="Actinide (n,\u03b3) loss")

        # Shade the fuel slab
        ax.axvspan(offset, offset + FUEL_SLAB_THICKNESS, color="gray", alpha=0.15,
                   label="Spent fuel slab")

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Depth into blanket [cm]")
        ax.set_ylabel("Net neutrons added (+) or lost (\u2212) per source neutron per bin")
        ax.set_title(f"Neutron balance vs depth \u2014 fuel slab midpoint = {midpt:.0f} cm")
        ax.legend(fontsize=8, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, TOTAL_LENGTH)

        plot_path = os.path.join(OUTPUT_DIR, f"neutron_balance_mid{midpt:.0f}cm.png")
        fig.savefig(plot_path, dpi=200)
        print(f"Neutron balance plot saved to: {plot_path}")
        plt.close(fig)


def plot_cumulative_neutron_balance(all_results):
    """
    Plot the *cumulative* neutron balance vs depth: running integral of
    gains and losses from the source face (z=0) to each depth bin.

    At any depth d the cumulative value tells you the total neutrons
    gained or lost from z=0 to z=d per source neutron.
    """
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
    indices = np.linspace(0, n_cases - 1, min(4, n_cases), dtype=int)

    for ci in indices:
        r = fuel_results[ci]
        csv_path = r.get("nbal_csv")
        if csv_path is None or not os.path.exists(csv_path):
            continue

        data = _read_mesh_csv(csv_path)
        z = data.get("z_cm")
        if z is None:
            continue

        midpt = r["slab_midpoint_cm"]
        offset = r["slab_offset_cm"]

        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

        # --- Cumulative GAINS (positive) ---
        pb_n2n = np.cumsum(data.get("pb_n2n", np.zeros_like(z)))
        ax.plot(z, pb_n2n, color="tab:blue", linewidth=1.5, label="Pb (n,2n) gain")
        ax.fill_between(z, 0, pb_n2n, alpha=0.15, color="tab:blue")

        for nuc, color, ls in [("U235", "tab:red", "-"),
                                ("U238", "tab:orange", "-"),
                                ("Pu239", "tab:green", "-")]:
            fiss = data.get(f"{nuc}_fission", np.zeros_like(z))
            nu_f = data.get(f"{nuc}_nu_fission", np.zeros_like(z))
            net = np.cumsum(nu_f - fiss)
            ax.plot(z, net, color=color, linewidth=1.5, linestyle=ls,
                    label=f"{nuc} fission net gain")

        oa_fiss = data.get("other_act_fission", np.zeros_like(z))
        oa_nuf  = data.get("other_act_nu_fission", np.zeros_like(z))
        oa_net  = np.cumsum(oa_nuf - oa_fiss)
        if np.abs(oa_net).max() > 1e-15:
            ax.plot(z, oa_net, color="tab:olive", linewidth=1.2, linestyle="--",
                    label="Other actinide fission net")

        # --- Cumulative LOSSES (negative) ---
        nXt = np.cumsum(data.get("nXt", np.zeros_like(z)))
        ax.plot(z, -nXt, color="tab:purple", linewidth=1.5, label="(n,Xt) tritium loss")
        ax.fill_between(z, 0, -nXt, alpha=0.1, color="tab:purple")

        act_cap_raw = np.zeros_like(z)
        for nuc in KEY_ACTINIDES:
            act_cap_raw += data.get(f"{nuc}_ngamma", np.zeros_like(z))
        act_cap_raw += data.get("other_act_ngamma", np.zeros_like(z))
        act_cap = np.cumsum(act_cap_raw)
        ax.plot(z, -act_cap, color="tab:brown", linewidth=1.5,
                label="Actinide (n,\u03b3) loss")

        # --- Total cumulative neutron budget ---
        total = pb_n2n + np.cumsum(
            (data.get("U235_nu_fission", np.zeros_like(z)) - data.get("U235_fission", np.zeros_like(z)))
            + (data.get("U238_nu_fission", np.zeros_like(z)) - data.get("U238_fission", np.zeros_like(z)))
            + (data.get("Pu239_nu_fission", np.zeros_like(z)) - data.get("Pu239_fission", np.zeros_like(z)))
            + (oa_nuf - oa_fiss)
        ) - nXt - act_cap
        ax.plot(z, total, color="black", linewidth=2.0, linestyle=":",
                label="Net cumulative balance")

        # Shade the fuel slab
        ax.axvspan(offset, offset + FUEL_SLAB_THICKNESS, color="gray", alpha=0.15,
                   label="Spent fuel slab")

        ax.axhline(0, color="black", linewidth=0.4)
        ax.set_xlabel("Depth into blanket [cm]")
        ax.set_ylabel("Cumulative neutrons gained (+) or lost (\u2212)\nper source neutron")
        ax.set_title(f"Cumulative neutron balance \u2014 fuel slab midpoint = {midpt:.0f} cm")
        ax.legend(fontsize=8, loc="best", ncol=2)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, TOTAL_LENGTH)

        plot_path = os.path.join(OUTPUT_DIR, f"cumulative_nbal_mid{midpt:.0f}cm.png")
        fig.savefig(plot_path, dpi=200)
        print(f"Cumulative neutron balance plot saved to: {plot_path}")
        plt.close(fig)


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 72)
    print("  Sigma Pile Sensitivity Analysis")
    print("  Zap Energy Z-Pinch: Spent Fuel Slab in PbLi Blanket")
    print("=" * 72)
    print(f"\n  Total prism length:      {TOTAL_LENGTH} cm")
    print(f"  Cross section:           {CROSS_SECTION} x {CROSS_SECTION} cm")
    print(f"  Fuel slab thickness:     {FUEL_SLAB_THICKNESS} cm")
    print(f"  Number of positions:     {len(SLAB_MIDPOINTS)}")
    print(f"  Midpoint sweep range:    {SLAB_MIDPOINTS[0]:.1f} – {SLAB_MIDPOINTS[-1]:.1f} cm")
    print(f"  Particles per batch:     {PARTICLES}")
    print(f"  Batches:                 {BATCHES}")
    print(f"  Source energy:           14.1 MeV D-T fusion neutrons")
    print(f"  Mesh tally resolution:   {MESH_DZ*10:.0f} mm ({N_MESH_Z} bins along z)\n")

    # Build materials
    print("Building PbLi eutectic material (Li-17Pb-83)...")
    pbli_mat = build_pbli_material()

    print(f"Loading spent fuel inventory from '{CSV_PATH}'...")
    inventory = parse_inventory(CSV_PATH)
    print(f"  Loaded {len(inventory)} nuclides (noble gases excluded)")

    # Print the dominant nuclides
    total = sum(inventory.values())
    sorted_nucs = sorted(inventory.items(), key=lambda x: x[1], reverse=True)
    print("\n  Top 10 nuclides by atom fraction:")
    for nuc, atoms in sorted_nucs[:10]:
        print(f"    {nuc:12s}  {atoms/total*100:8.4f}%")

    fuel_mat = build_spent_fuel_material(inventory)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run parametric sweep
    all_results = []
    n_total = len(SLAB_MIDPOINTS) + 1  # +1 for the no-fuel baseline

    # --- Baseline: pure PbLi, no spent fuel ---
    print(f"\n{chr(9472)*60}")
    print(f"  Case 1/{n_total}: NO FUEL (pure PbLi baseline)")
    print(f"{chr(9472)*60}")
    result = run_case(None, pbli_mat, fuel_mat, 0)
    all_results.append(result)
    print(f"  TBR           = {result.get('TBR_mean', 0):.6f} +/- {result.get('TBR_std', 0):.6f}")
    print(f"  (n,2n) rate   = {result.get('n2n_mean', 0):.6e} +/- {result.get('n2n_std', 0):.6e}")
    print(f"  Leakage       = {result['leakage_mean']:.6e} +/- {result['leakage_std']:.6e}")

    # --- Sweep spent fuel slab positions ---
    for i, midpt in enumerate(SLAB_MIDPOINTS):
        print(f"\n{chr(9472)*60}")
        print(f"  Case {i+2}/{n_total}: slab midpoint = {midpt:.1f} cm")
        print(f"{chr(9472)*60}")

        result = run_case(midpt, pbli_mat, fuel_mat, i + 1)
        all_results.append(result)

        # Print summary for this case
        print(f"  TBR           = {result.get('TBR_mean', 0):.6f} +/- {result.get('TBR_std', 0):.6f}")
        print(f"  Fission rate  = {result['fission_mean']:.6e} ± {result['fission_std']:.6e}")
        print(f"  Capture rate  = {result['capture_mean']:.6e} ± {result['capture_std']:.6e}")
        print(f"  (n,2n) rate   = {result.get('n2n_mean', 0):.6e} ± {result.get('n2n_std', 0):.6e}")
        print(f"  Leakage       = {result['leakage_mean']:.6e} ± {result['leakage_std']:.6e}")
        if "mesh_csv" in result:
            print(f"  Mesh CSV      = {result['mesh_csv']}")

    # Save all results as JSON
    json_path = os.path.join(OUTPUT_DIR, "all_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to: {json_path}")

    # Generate plots
    plot_results(all_results)
    plot_mesh_profiles(all_results)
    plot_neutron_balance(all_results)
    plot_cumulative_neutron_balance(all_results)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Midpoint [cm]':>14}  {'TBR':>12}  {'Fission':>12}  {'Capture':>12}  "
          f"{'(n,2n)':>12}  {'Leakage':>12}")
    print("-" * 90)
    for r in all_results:
        mid = r['slab_midpoint_cm']
        label = f"{mid:14.1f}" if mid is not None else "    (no fuel) "
        print(f"{label}  "
              f"{r.get('TBR_mean',0):12.6f}  "
              f"{r['fission_mean']:12.6e}  "
              f"{r['capture_mean']:12.6e}  "
              f"{r.get('n2n_mean',0):12.6e}  "
              f"{r['leakage_mean']:12.6e}")
    print("=" * 90)
    print("\nDone.")


if __name__ == "__main__":
    main()