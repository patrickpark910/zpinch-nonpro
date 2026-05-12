"""
OpenMC Pin Cell Depletion Model — 4S Reactor (U-10Zr Fuel)
==========================================================

Models a single fuel pin in an infinite lattice representative of one
fuel assembly in the Toshiba 4S (Super-Safe, Small and Simple)
sodium-cooled fast microreactor.  Power is normalised to one of the
18 assemblies (169 pins each).  The 4S delivers 10 MWe (~30 MWth)
over a 30-year core life without refueling, using metallic U-10wt%Zr
fuel enriched to <20% U-235 (HALEU), sodium bonding, and HT-9
ferritic-martensitic steel cladding.

Geometry and conditions from Toshiba NRC Pre-Application Review
(PSN-2008-0045, February 21, 2008) and ANL fuel presentation:

* Fuel slug diameter         : 10.4  mm  → radius 0.52 cm
* Cladding outer diameter    : 14.0  mm  → radius 0.70 cm
* Cladding thickness         : 1.1   mm  → clad ID radius 0.59 cm
* Na bond gap                : slug OD to clad ID (0.52–0.59 cm)
* Smear density (BOC)        : 78%
* Fuel alloy                 : U-10 wt% Zr, ρ ≈ 15.85 g/cm³ (at 78% SD → ~12.36)
* 235-U enrichment           : 17% inner core / 19% outer core
* Cladding material          : HT-9 ferritic-martensitic steel
* Coolant                    : liquid sodium, inlet/outlet 355/510 °C
* Core thermal power         : 30 MWth  (10 MWe)
* Core height                : 2.5 m
* 18 fuel subassemblies × 169 pins/assembly = 3,042 fuel pins
* Average/peak burnup        : 34,000 / 55,000 MWd/t
* Peak cladding hotspot      : 609 °C  (~882 K)
* Max. linear power          : 8 kW/m
* Plenum/fuel volume ratio   : 1.3
* No on-site refueling for 30 years

Adjust parameters in the "USER-EDITABLE PARAMETERS" section as needed.
"""

import os
import numpy as np
import openmc
import openmc.deplete

# ============================================================
# USER-EDITABLE PARAMETERS
# ============================================================

# --- Geometry (cm) — from Toshiba PSN-2008-0045 ---
fuel_or      = 0.520       # fuel slug outer radius  (10.4 mm diameter)
bond_or      = 0.590       # Na bond gap outer radius = cladding inner radius
                           #   clad ID = clad OD - 2 × thickness = 14.0 - 2×1.1 = 11.8 mm
clad_or      = 0.700       # cladding outer radius    (14.0 mm diameter)
pin_pitch    = 1.680       # hex lattice pitch estimate (P/D ≈ 1.2 for SFR)

# --- Fuel (Toshiba spec: inner 17%, outer 19%) ---
# Model the OUTER core pin (bounding enrichment); switch to 0.17 for inner.
enrich_u235  = 0.18        # U-235 enrichment (weight fraction), outer core
wt_frac_zr   = 0.10        # Zr weight fraction in U-10Zr
# NOTE: 78% smear density is geometric (fuel_slug_area / clad_ID_area),
# already captured by the fuel slug and bond gap radii.  The slug itself
# is cast at near-theoretical density.
fuel_density = 15.85       # g/cm³  (as-cast U-10Zr theoretical density)
fuel_temp    = 900.0       # fuel average temperature (K), centerline < 630°C per spec

# --- Cladding (HT-9: Fe-12Cr-1Mo-0.5W-0.3V-0.2C approx.) ---
clad_density = 7.87        # g/cm³
clad_temp    = 840.0       # K  (nominal peak ~567°C; hotspot 609°C = 882 K)

# --- Coolant (liquid sodium, inlet 355°C / outlet 510°C) ---
na_temp      = 705.0       # K  (average coolant temp ≈ 432°C)
na_density   = 0.847       # g/cm³ at ~432 °C

# --- Reactor power & depletion schedule ---
# Model one fuel assembly in an infinite lattice.
# Total core power split equally among 18 assemblies.
n_assemblies    = 18
pins_per_assy   = 169
core_power      = 30.0e6                          # total core thermal power [W]
assy_power      = core_power / n_assemblies        # W per assembly ≈ 1.667 MW
pin_power       = assy_power / pins_per_assy       # W per pin
pin_length      = 250.0                            # active fuel length [cm] = 2.5 m

# Depletion time steps (days).  30 years ≈ 10,957 days.
# Use shorter steps early (high reactivity swing), then lengthen.
days_per_year = 365.25
total_years   =  30.00

# Build a schedule: 1 step of 60d (initial transient), then 5-year steps
depletion_days = (
    [15.0]    * 2  +     #    60 d  (initial transient)
    [5*365.25] * 5  )    # 9131 d  (5 steps × 5 yr = years 1–26)

# Append remainder to reach 30 years exactly
_scheduled = sum(depletion_days)
_target_days = total_years * days_per_year
_remainder = _target_days - _scheduled
if _remainder > 0:
    depletion_days.append(_remainder)
# Total ≈ 10,957 d = 30 yr

# --- Monte Carlo settings ---
batches      = 120
inactive     =  20
particles    = int(1e4)       # increase for production runs (≥50 000)

# --- Depletion chain ---
chain_file = "/home/ppark/openmc/data/chain_endfb81_fast.xml"

# --- Output directories ---
OPENMC_DIR   = "uzr_openmc"       # XMLs, H5s, statepoints, tallies
RESULTS_DIR  = "./"      # figures, CSVs

# --- Output file ---
DEPLETE_FILE = os.path.join(OPENMC_DIR, "depletion_results.h5")


# ============================================================
# FUNCTION 1: BUILD MODEL AND WRITE model.xml
# ============================================================

def build_model():
    """Construct the OpenMC model and export everything to a single model.xml.

    Returns
    -------
    model : openmc.Model
        The fully-defined model object.
    fuel_mat : openmc.Material
        Reference to the depletable fuel material (needed for post-processing).
    """

    # --- Fuel: U-10Zr ---
    fuel = openmc.Material(material_id=100, name="U-10Zr fuel")
    fuel.set_density("g/cm3", fuel_density)
    fuel.temperature = fuel_temp

    wt_frac_u = 1.0 - wt_frac_zr
    fuel.add_nuclide("U235", wt_frac_u * enrich_u235,         percent_type="wo")
    fuel.add_nuclide("U238", wt_frac_u * (1.0 - enrich_u235), percent_type="wo")
    fuel.add_element("Zr", wt_frac_zr, percent_type="wo")

    fuel.depletable = True
    fuel.volume = np.pi * fuel_or**2 * pin_length   # cm³ for one pin

    # --- Sodium bond (same Na, in fuel-clad gap) ---
    na_bond = openmc.Material(material_id=200, name="Na bond")
    na_bond.set_density("g/cm3", na_density)
    na_bond.temperature = fuel_temp
    na_bond.add_element("Na", 1.0)

    # --- Cladding: HT-9 (simplified) ---
    clad = openmc.Material(material_id=300, name="HT-9 cladding")
    clad.set_density("g/cm3", clad_density)
    clad.temperature = clad_temp
    clad.add_element("Fe", 0.845, percent_type="wo")
    clad.add_element("Cr", 0.120, percent_type="wo")
    clad.add_element("Mo", 0.010, percent_type="wo")
    clad.add_element("W",  0.005, percent_type="wo")
    clad.add_element("V",  0.003, percent_type="wo")
    clad.add_element("Mn", 0.006, percent_type="wo")
    clad.add_element("Si", 0.004, percent_type="wo")
    clad.add_element("Ni", 0.005, percent_type="wo")
    clad.add_element("C",  0.002, percent_type="wo")

    # --- Coolant: liquid sodium ---
    coolant = openmc.Material(material_id=400, name="Na coolant")
    coolant.set_density("g/cm3", na_density)
    coolant.temperature = na_temp
    coolant.add_element("Na", 1.0)

    materials = openmc.Materials([fuel, na_bond, clad, coolant])

    # ---- Geometry — hexagonal pin-cell with reflective boundaries ----

    fuel_surface = openmc.ZCylinder(surface_id=1, r=fuel_or)
    bond_surface = openmc.ZCylinder(surface_id=2, r=bond_or)
    clad_surface = openmc.ZCylinder(surface_id=3, r=clad_or)

    min_z = openmc.ZPlane(surface_id=4, z0=-pin_length / 2, boundary_type="reflective")
    max_z = openmc.ZPlane(surface_id=5, z0=pin_length / 2, boundary_type="reflective")

    hex_prism = openmc.model.HexagonalPrism(
        edge_length=pin_pitch / np.sqrt(3.0),
        orientation="y",
        boundary_type="reflective",
    )

    # Update your cells to be bounded by the Z-planes
    fuel_cell = openmc.Cell(cell_id=10, name="fuel", fill=fuel, region=-fuel_surface & +min_z & -max_z)
    bond_cell = openmc.Cell(cell_id=20, name="bond", fill=na_bond, region=+fuel_surface & -bond_surface & +min_z & -max_z)
    clad_cell = openmc.Cell(cell_id=30, name="clad", fill=clad, region=+bond_surface & -clad_surface & +min_z & -max_z)
    coolant_cell = openmc.Cell(cell_id=40, name="coolant", fill=coolant, region=+clad_surface & -hex_prism & +min_z & -max_z)

    root_universe = openmc.Universe(universe_id=1, cells=[fuel_cell, bond_cell, clad_cell, coolant_cell])
    geometry = openmc.Geometry(root_universe)

    # ---- Settings ----

    settings = openmc.Settings()
    settings.batches   = batches
    settings.inactive  = inactive
    settings.particles = particles
    settings.temperature = {"method": "interpolation",}

    r_src     = openmc.stats.Uniform(0.0, fuel_or)
    theta_src = openmc.stats.Uniform(0.0, 2 * np.pi)
    z_src     = openmc.stats.Uniform(-pin_length / 2, pin_length / 2)
    src_spatial = openmc.stats.CylindricalIndependent(r_src, theta_src, z_src)
    settings.source = openmc.IndependentSource(space=src_spatial)

    # ---- Tallies (optional — k-eff is tracked automatically) ----

    tallies = openmc.Tallies()

    energy_filter = openmc.EnergyFilter(np.logspace(np.log10(1e-5), np.log10(20e6), 201))
    cell_filter   = openmc.CellFilter(fuel_cell)

    flux_tally = openmc.Tally(tally_id=1, name="fuel flux spectrum")
    flux_tally.filters = [cell_filter, energy_filter]
    flux_tally.scores  = ["flux"]
    tallies.append(flux_tally)

    fission_tally = openmc.Tally(tally_id=2, name="fission rate")
    fission_tally.filters = [cell_filter]
    fission_tally.scores  = ["fission"]
    tallies.append(fission_tally)

    # ---- Assemble model and write single model.xml ----

    model = openmc.Model(geometry, materials, settings, tallies)
    # model.export_to_model_xml() # don't export for depletion calcs
    # print("Wrote model.xml")

    return model, fuel


# ============================================================
# FUNCTION 2: RUN DEPLETION
# ============================================================

def run_depletion(model):
    """Set up the depletion operator and integrator, then run.

    Parameters
    ----------
    model : openmc.Model
        The OpenMC model (already exported to model.xml).
    """

    print("=" * 60)
    print("4S Pin-Cell Depletion Model (Toshiba PSN-2008-0045)")
    print("=" * 60)
    print(f"  Fuel          : U-{wt_frac_zr*100:.0f}Zr, {enrich_u235*100:.0f}% U-235 (outer core)")
    print(f"  Fuel slug OD  : {fuel_or*20:.1f} mm")
    print(f"  Clad OD / t   : {clad_or*20:.1f} mm / {(clad_or - bond_or)*10:.1f} mm")
    print(f"  Smear density : {(fuel_or/bond_or)**2*100:.0f}%")
    print(f"  Fuel density  : {fuel_density} g/cm³ (theoretical)")
    print(f"  Cladding      : HT-9")
    print(f"  Coolant       : Na, inlet/outlet 355/510 °C")
    print(f"  Pin power     : {pin_power:.1f} W  ({pin_power/pin_length:.1f} W/cm)")
    print(f"  Assembly      : {pins_per_assy} pins  (1 of {n_assemblies} in core)")
    print(f"  Assy power    : {assy_power/1e6:.3f} MWth")
    print(f"  Depletion     : {sum(depletion_days)/days_per_year:.1f} years "
          f"in {len(depletion_days)} steps")
    print(f"  MC particles  : {particles} / batch × {batches} batches")
    print("=" * 60)

    # Run depletion from the openmc output directory so all
    # intermediate files (XMLs, H5s, statepoints) land there.
    orig_dir = os.getcwd()
    os.chdir(OPENMC_DIR)

    operator = openmc.deplete.CoupledOperator(
        model,
        chain_file=chain_file,
        normalization_mode="fission-q",
    )

    integrator = openmc.deplete.CECMIntegrator(
        operator,
        depletion_days,
        power=pin_power,
        timestep_units="d",
    )

    integrator.integrate()

    os.chdir(orig_dir)
    print("\nDepletion calculation complete.")


# ============================================================
# FUNCTION 3: POST-PROCESS DEPLETION RESULTS
# ============================================================

def post_process(fuel_mat):
    results = openmc.deplete.Results(DEPLETE_FILE)

    # --- k-eff vs. time ---
    time_steps, keffs = results.get_keff()
    time_years = time_steps / (days_per_year * 86400)

    # --- Find material and nuclide indices from first step ---
    r0 = results[0]
    print("Material index map:", r0.index_mat)

    # Get the integer index for our fuel material (only 1 depletable mat)
    mat_idx = list(r0.index_mat.values())[0]

    nuclides_of_interest = [
        "U235", "U238", "Pu239", "Pu240", "Pu241",
        "Zr93", "Cs137", "Sr90", "Xe135",
    ]

    nuclides_tracked = [
        # Actinides
        "U234", "U235", "U236", "U238",
        "Np237",
        "Pu238", "Pu239", "Pu240", "Pu241", "Pu242",
        "Am241", "Am242_m1", "Am243",
        "Cm242", "Cm243", "Cm244", "Cm245",
        # Zirconium matrix
        "Zr90", "Zr91", "Zr92", "Zr93", "Zr94", "Zr95", "Zr96",
        # High-absorption fission products
        "Sm149", "Sm151", "Gd155", "Gd157",
        "Rh103", "Nd143", "Nd145", "Cs133",
        "Eu151", "Eu153", "Ag109",
        # High-inventory fission products
        "Mo95", "Mo97", "Mo98", "Mo100",
        "Xe126", "Xe128", "Xe130", "Xe131", "Xe132", "Xe134", "Xe136",
        "Kr80", "Kr82", "Kr83", "Kr84", "Kr86",
        "Cs135", "Cs137",
        "Ba138", "La139", "Ce140", "Ce142",
        "Tc99", "Pr141", "Ru101", "Ru102",
        "Nd144", "Nd146", "Y89", "Sr88", "Sr90",
    ]

    # --- Combined table: k-eff + U235 at each step ---
    print("\n  Time [yr]     k-eff       ± σ          U235 [atoms]")
    print("  " + "-" * 58)
    u235_idx = r0.index_nuc["U235"]
    for i, (t, (k, sig)) in enumerate(zip(time_years, keffs)):
        step = results[i]
        u235 = step.data[mat_idx, u235_idx]
        print(f"  {t:8.2f}     {k:.5f}   ± {sig:.5f}    {u235:.4e}")

    # --- EOL inventories ---
    print("\n  Nuclide inventories at end of life (atoms):")
    print("  " + "-" * 44)
    last = results[-1]
    for nuc in nuclides_of_interest:
        if nuc in last.index_nuc:
            nuc_idx = last.index_nuc[nuc]
            print(f"  {nuc:8s}  {last.data[mat_idx, nuc_idx]:.4e}")
        else:
            print(f"  {nuc:8s}  (not in chain)")

    # --- Tracked nuclides for Z-pinch blanket ---
    print("\n  Tracked nuclides for Z-pinch spent fuel (atoms):")
    print("  " + "-" * 44)
    last = results[-1]
    for nuc in nuclides_tracked:
        if nuc in last.index_nuc:
            nuc_idx = last.index_nuc[nuc]
            val = last.data[mat_idx, nuc_idx]
            print(f"  {nuc:10s}  {val:.4e}")
        else:
            print(f"  {nuc:10s}  (not in chain)")

    last = results[-1]
    mat_idx = list(last.index_mat.values())[0]
    inventory = [(nuc, last.data[mat_idx, idx]) for nuc, idx in last.index_nuc.items()]
    inventory.sort(key=lambda x: x[1], reverse=True)

    # print("\n  All nuclide inventories at EOL (atoms), descending:")
    # print("  " + "-" * 44)
    # for nuc, val in inventory:
    #     if val > 0:
    #         print(f"  {nuc:8s}  {val:.4e}")

    print("\n  Saving EOL inventory to CSV file...")
    import csv
    csv_path = os.path.join(RESULTS_DIR, "uzr_eol_inventory.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["nuclide", "atoms"])
        for nuc, val in inventory:
            if val > 0:
                writer.writerow([nuc, f"{val:.6e}"])
    print(f"  Saved to {csv_path}")

    # --- Save tracked nuclides as separate CSV for Z-pinch input ---
    print("\n  Saving tracked nuclide inventory to CSV file...")
    csv_tracked_path = os.path.join(RESULTS_DIR, "uzr_eol_inventory_tracked.csv")
    with open(csv_tracked_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["nuclide", "atoms"])
        for nuc in nuclides_tracked:
            if nuc in last.index_nuc:
                nuc_idx = last.index_nuc[nuc]
                val = last.data[mat_idx, nuc_idx]
                if val > 0:
                    writer.writerow([nuc, f"{val:.6e}"])
    print(f"  Saved to {csv_tracked_path}")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(time_years, keffs[:, 0], yerr=keffs[:, 1],
                    fmt="o-", markersize=3, capsize=2, label=r"$k_\mathrm{eff}$")
        ax.axhline(1.0, color="grey", ls="--", lw=0.8)
        ax.set_xlabel("Irradiation time [years]")
        ax.set_ylabel(r"$k_\mathrm{eff}$")
        ax.set_title("4S Pin-Cell Depletion — U-10Zr / Na / HT-9")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, "uzr_keff_vs_time.png")
        fig.savefig(plot_path, dpi=150)
        print(f"\n  Plot saved to {plot_path}")
    except ImportError:
        print("\n  matplotlib not available — skipping plot.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Create output directories
    os.makedirs(OPENMC_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Always build the model (needed for fuel material reference)
    model, fuel_mat = build_model()

    if os.path.isfile(DEPLETE_FILE):
        print(f"\n{DEPLETE_FILE} already exists--skipping depletion, jumping to post-processing.\n")
    else:
        run_depletion(model)

    post_process(fuel_mat)
