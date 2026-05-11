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

# Slab position offsets from the source face [cm]
# Sweep from 0 to (TOTAL_LENGTH - FUEL_SLAB_THICKNESS) in steps
SLAB_POSITIONS = np.arange(0.0, TOTAL_LENGTH - FUEL_SLAB_THICKNESS + 1.0, 5.0)

MESH_DZ = 0.1                                 # Mesh tally z-resolution [cm] (1 mm)
N_MESH_Z = int(TOTAL_LENGTH / MESH_DZ)       # Number of mesh bins along z (1500)

BATCHES = 50                                  # Number of batches
INACTIVE_BATCHES = 10                         # Inactive batches (for fixed-source, set to 0)
PARTICLES = 50000                             # Particles per batch

# PbLi eutectic parameters (Li-17Pb-83 by atom fraction)
PBLI_LI_ATOM_FRAC = 0.17                     # 17 at% Li (natural: 7.5% Li-6, 92.5% Li-7)
PBLI_PB_ATOM_FRAC = 0.83                     # 83 at% Pb (natural isotopic mix)
PBLI_DENSITY = 9.49                           # g/cm³ at ~500°C operating temperature

# Spent fuel density — U-10Zr metallic alloy (consistent with ZFFR blanket design)
SPENT_FUEL_DENSITY = 15.8                     # g/cm³

OUTPUT_DIR = "sigma_pile_results"


# ==============================================================================
# Parse spent fuel inventory from CSV
# ==============================================================================

def parse_inventory(csv_path):
    """
    Read the BOL inventory CSV and return a dict of {openmc_nuclide_name: atom_count}.
    Converts CSV naming (e.g. 'Am242_m1') to OpenMC naming (e.g. 'Am242_m1').
    Skips noble gases (He, Kr, Xe) as they would escape metallic fuel.
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

            # OpenMC uses the same naming convention: Element + mass number
            # Metastable states: CSV has 'Am242_m1' -> OpenMC uses 'Am242_m1'
            inventory[raw_name] = atoms

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

def run_case(slab_offset, pbli_mat, fuel_mat, run_index):
    """
    Build and run an OpenMC model for the spent fuel slab at a given offset
    from the source face.

    Parameters
    ----------
    slab_offset : float
        Distance [cm] from the source face to the front of the spent fuel slab.
    pbli_mat : openmc.Material
        The PbLi blanket material.
    fuel_mat : openmc.Material
        The spent fuel material.
    run_index : int
        Index for the parametric sweep (used for directory naming).

    Returns
    -------
    dict : Tally results for this case.
    """

    case_dir = os.path.join(OUTPUT_DIR, f"case_{run_index:03d}_offset_{slab_offset:.1f}cm")
    os.makedirs(case_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------------
    # Clone materials so each case is independent
    pbli_front = pbli_mat.clone()
    pbli_front.name = "PbLi_front"
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
    z_source     = openmc.ZPlane(z0=0.0,                        boundary_type="vacuum")
    z_fuel_front = openmc.ZPlane(z0=slab_offset)
    z_fuel_back  = openmc.ZPlane(z0=slab_offset + FUEL_SLAB_THICKNESS)
    z_end        = openmc.ZPlane(z0=TOTAL_LENGTH,               boundary_type="vacuum")

    # Lateral surfaces (reflective to approximate infinite slab)
    x_min = openmc.XPlane(x0=-half_w, boundary_type="reflective")
    x_max = openmc.XPlane(x0=+half_w, boundary_type="reflective")
    y_min = openmc.YPlane(y0=-half_w, boundary_type="reflective")
    y_max = openmc.YPlane(y0=+half_w, boundary_type="reflective")

    lateral = +x_min & -x_max & +y_min & -y_max

    # Regions
    has_front = slab_offset > 0.0
    has_back  = (slab_offset + FUEL_SLAB_THICKNESS) < TOTAL_LENGTH

    cells = []

    # Front PbLi region (may have zero thickness if slab_offset == 0)
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

    # --- Tally 2: Fission rate in spent fuel ---
    fission_tally = openmc.Tally(name="fission_rate")
    fission_tally.filters = [openmc.MaterialFilter([fuel])]
    fission_tally.scores = ["fission"]
    tallies.append(fission_tally)

    # --- Tally 3: Radiative capture in spent fuel ---
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

    # --- Tally 5: Neutron flux spectrum in spent fuel (lethargy groups) ---
    energy_bins = np.logspace(np.log10(1e-5), np.log10(20e6), 726)  # 725 groups
    energy_filter = openmc.EnergyFilter(energy_bins)
    spectrum_tally = openmc.Tally(name="fuel_spectrum")
    spectrum_tally.filters = [openmc.MaterialFilter([fuel]), energy_filter]
    spectrum_tally.scores = ["flux"]
    tallies.append(spectrum_tally)

    # --- Tally 6: Heating in each region ---
    all_mats = pbli_filter_mats + [fuel]
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

    # --- Tally 8: Absorption in spent fuel (total) ---
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
            openmc.run(output=False)

        # Extract results
        sp_file = openmc.StatePoint(sp_filename)
        results = {"slab_offset_cm": slab_offset}

        # TBR
        if pbli_filter_mats:
            t = sp_file.get_tally(name="TBR")
            results["TBR_mean"] = float(t.mean.sum())
            results["TBR_std"]  = float(np.sqrt((t.std_dev**2).sum()))

        # Fission rate
        t = sp_file.get_tally(name="fission_rate")
        results["fission_mean"] = float(t.mean.sum())
        results["fission_std"]  = float(np.sqrt((t.std_dev**2).sum()))

        # Capture rate
        t = sp_file.get_tally(name="capture_rate")
        results["capture_mean"] = float(t.mean.sum())
        results["capture_std"]  = float(np.sqrt((t.std_dev**2).sum()))

        # (n,2n) rate
        if pbli_filter_mats:
            t = sp_file.get_tally(name="n2n_rate")
            results["n2n_mean"] = float(t.mean.sum())
            results["n2n_std"]  = float(np.sqrt((t.std_dev**2).sum()))

        # Neutron spectrum in fuel — save full array
        t = sp_file.get_tally(name="fuel_spectrum")
        results["fuel_spectrum_mean"] = t.mean.flatten().tolist()
        results["fuel_spectrum_std"]  = t.std_dev.flatten().tolist()
        results["energy_bins_eV"]     = energy_bins.tolist()

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
        t = sp_file.get_tally(name="absorption_fuel")
        results["absorption_mean"] = float(t.mean.sum())
        results["absorption_std"]  = float(np.sqrt((t.std_dev**2).sum()))

        # -----------------------------------------------------------
        # Extract depth-resolved mesh tallies
        # -----------------------------------------------------------
        z_edges = np.linspace(0.0, TOTAL_LENGTH, N_MESH_Z + 1)
        z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

        # Volume mesh tally (8 scores)
        mt = sp_file.get_tally(name="mesh_depth_profile")
        # mt.mean has shape (N_MESH_Z, len(MESH_SCORES), 1) after reshaping
        # Use get_values for clean extraction per score
        mesh_data = {"z_cm": z_centers}
        for si, score_name in enumerate(MESH_SCORES):
            # Slice the tally: the mesh filter produces N_MESH_Z bins,
            # and each score is a separate column in the flattened array.
            # With shape (n_mesh_bins, n_scores, 1):
            mean_vals = mt.mean[:, si, 0] if mt.mean.ndim == 3 else mt.mean.flatten()
            std_vals  = mt.std_dev[:, si, 0] if mt.std_dev.ndim == 3 else mt.std_dev.flatten()

            # Sanitise score name for use as a column header
            col = score_name.replace("(", "").replace(")", "").replace(",", "").replace("-", "_")
            mesh_data[f"{col}_mean"] = mean_vals
            mesh_data[f"{col}_std"]  = std_vals

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

    offsets      = [r["slab_offset_cm"] for r in all_results]
    tbr_mean     = [r.get("TBR_mean", 0.0) for r in all_results]
    tbr_std      = [r.get("TBR_std", 0.0)  for r in all_results]
    fiss_mean    = [r["fission_mean"]       for r in all_results]
    fiss_std     = [r["fission_std"]        for r in all_results]
    cap_mean     = [r["capture_mean"]       for r in all_results]
    cap_std      = [r["capture_std"]        for r in all_results]
    n2n_mean     = [r.get("n2n_mean", 0.0)  for r in all_results]
    n2n_std      = [r.get("n2n_std", 0.0)   for r in all_results]
    leak_mean    = [r["leakage_mean"]       for r in all_results]
    leak_std     = [r["leakage_std"]        for r in all_results]
    abs_mean     = [r["absorption_mean"]    for r in all_results]
    abs_std      = [r["absorption_std"]     for r in all_results]

    fig, axes = plt.subplots(3, 2, figsize=(14, 16), constrained_layout=True)
    fig.suptitle(
        "Sigma Pile Sensitivity: Spent Fuel Slab Position in Zap Energy Z-Pinch PbLi Blanket",
        fontsize=14, fontweight="bold"
    )

    # (a) Tritium Breeding Rate
    ax = axes[0, 0]
    ax.errorbar(offsets, tbr_mean, yerr=tbr_std, fmt="o-", capsize=3, color="tab:blue")
    ax.set_xlabel("Spent Fuel Slab Offset from Source [cm]")
    ax.set_ylabel("Tritium Production Rate\n[per source neutron]")
    ax.set_title("(a) Tritium Breeding Rate in PbLi")
    ax.grid(True, alpha=0.3)

    # (b) Fission Rate in Fuel
    ax = axes[0, 1]
    ax.errorbar(offsets, fiss_mean, yerr=fiss_std, fmt="s-", capsize=3, color="tab:red")
    ax.set_xlabel("Spent Fuel Slab Offset from Source [cm]")
    ax.set_ylabel("Fission Rate\n[per source neutron]")
    ax.set_title("(b) Fission Rate in Spent Fuel")
    ax.grid(True, alpha=0.3)

    # (c) Capture Rate in Fuel
    ax = axes[1, 0]
    ax.errorbar(offsets, cap_mean, yerr=cap_std, fmt="^-", capsize=3, color="tab:green")
    ax.set_xlabel("Spent Fuel Slab Offset from Source [cm]")
    ax.set_ylabel("(n,γ) Capture Rate\n[per source neutron]")
    ax.set_title("(c) Radiative Capture in Spent Fuel")
    ax.grid(True, alpha=0.3)

    # (d) (n,2n) Rate in PbLi
    ax = axes[1, 1]
    ax.errorbar(offsets, n2n_mean, yerr=n2n_std, fmt="D-", capsize=3, color="tab:orange")
    ax.set_xlabel("Spent Fuel Slab Offset from Source [cm]")
    ax.set_ylabel("(n,2n) Rate in PbLi\n[per source neutron]")
    ax.set_title("(d) Pb Neutron Multiplication")
    ax.grid(True, alpha=0.3)

    # (e) Neutron Leakage Through Back Face
    ax = axes[2, 0]
    ax.errorbar(offsets, leak_mean, yerr=leak_std, fmt="v-", capsize=3, color="tab:purple")
    ax.set_xlabel("Spent Fuel Slab Offset from Source [cm]")
    ax.set_ylabel("Neutron Current (Leakage)\n[per source neutron]")
    ax.set_title("(e) Leakage Through Back Face")
    ax.grid(True, alpha=0.3)

    # (f) Total Absorption in Fuel
    ax = axes[2, 1]
    ax.errorbar(offsets, abs_mean, yerr=abs_std, fmt="p-", capsize=3, color="tab:brown")
    ax.set_xlabel("Spent Fuel Slab Offset from Source [cm]")
    ax.set_ylabel("Total Absorption in Fuel\n[per source neutron]")
    ax.set_title("(f) Total Neutron Absorption in Spent Fuel")
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(OUTPUT_DIR, "sigma_pile_sensitivity.png")
    fig.savefig(plot_path, dpi=200)
    print(f"\nPlots saved to: {plot_path}")
    plt.close(fig)

    # --- Additional plot: select neutron spectra at a few slab positions ---
    fig2, ax2 = plt.subplots(figsize=(10, 6), constrained_layout=True)
    indices_to_plot = np.linspace(0, len(all_results) - 1, min(6, len(all_results)), dtype=int)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(indices_to_plot)))

    for color_idx, i in enumerate(indices_to_plot):
        r = all_results[i]
        e_bins = np.array(r["energy_bins_eV"])
        e_mid = 0.5 * (e_bins[:-1] + e_bins[1:])
        lethargy_width = np.log(e_bins[1:] / e_bins[:-1])
        flux = np.array(r["fuel_spectrum_mean"])
        flux_per_lethargy = flux / lethargy_width
        ax2.loglog(e_mid, flux_per_lethargy, color=cmap[color_idx],
                   label=f"d = {r['slab_offset_cm']:.0f} cm", alpha=0.8)

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

    # Pick a subset of cases to overlay (avoid clutter)
    n_cases = len(all_results)
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
            r = all_results[case_i]
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
            ax.semilogy(z, y, color=cmap[ci], alpha=0.75, linewidth=0.8,
                        label=f"d = {offset:.0f} cm")

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
        r = all_results[case_i]
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
        ax3.semilogy(z, j, color=cmap[ci], alpha=0.75, linewidth=0.8,
                     label=f"d = {offset:.0f} cm")
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
    print(f"  Number of positions:     {len(SLAB_POSITIONS)}")
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
    for i, offset in enumerate(SLAB_POSITIONS):
        print(f"\n{'─'*60}")
        print(f"  Case {i+1}/{len(SLAB_POSITIONS)}: slab offset = {offset:.1f} cm")
        print(f"{'─'*60}")

        result = run_case(offset, pbli_mat, fuel_mat, i)
        all_results.append(result)

        # Print summary for this case
        print(f"  TBR           = {result.get('TBR_mean', 0):.6f} ± {result.get('TBR_std', 0):.6f}")
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

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Offset [cm]':>12}  {'TBR':>12}  {'Fission':>12}  {'Capture':>12}  "
          f"{'(n,2n)':>12}  {'Leakage':>12}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['slab_offset_cm']:12.1f}  "
              f"{r.get('TBR_mean',0):12.6f}  "
              f"{r['fission_mean']:12.6e}  "
              f"{r['capture_mean']:12.6e}  "
              f"{r.get('n2n_mean',0):12.6e}  "
              f"{r['leakage_mean']:12.6e}")
    print("=" * 90)
    print("\nDone.")


if __name__ == "__main__":
    main()