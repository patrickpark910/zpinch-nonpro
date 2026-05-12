"""
OpenMC Pin Cell Depletion Model — AP1000 Reactor (UO₂ Fuel)
============================================================

Models a single fuel pin in an infinite lattice representative of one
fuel rod in the Westinghouse AP1000 pressurised-water reactor.
Power is normalised to the average rod.  The AP1000 delivers 3400 MWth
(~1117 MWe net) using sintered UO₂ pellets in ZIRLO cladding, cooled
by pressurised light water at 15.5 MPa.

Geometry and conditions from the NRC AP1000 Design Control Document
(ML071580895) and the Westinghouse reactor comparison data sheet:

* Fuel pellet diameter       : 0.81915 cm  → radius 0.409575 cm
* Cladding outer diameter    : 0.950   cm  → radius 0.4750   cm
* Cladding thickness         : 0.05715 cm  → clad ID radius  0.41785 cm
* Diametral gap (He-filled)  : 0.01651 cm  (pellet-to-clad-ID)
* Fuel assembly              : 17×17 XL Robust Fuel Assembly (RFA)
* UO₂ rods per assembly      : 264
* Fuel assemblies in core    : 157
* Total fuel rods            : 41,448
* UO₂ pellet density         : 95.5% of theoretical (10.97 g/cm³)
* Active fuel height         : 426.72 cm   (14 ft)
* Rod pitch                  : 1.26   cm   (square lattice)
* Cladding material          : ZIRLO  (Zr-1Nb-1Sn-0.1Fe)
* Gap fill                   : Helium
* Coolant                    : Light water, 15.5 MPa
*   Inlet / Outlet temp      : ~280 / ~327 °C
*   Average coolant temp      : ~303 °C  (576 K)
* Core thermal power         : 3400 MWth
* 1st-cycle enrichments      : 2.35 / 3.40 / 4.45 wt% (3 regions)
* Target discharge burnup    : 60 GWd/MTU

Adjust parameters in the "USER-EDITABLE PARAMETERS" section as needed.
"""

import os
import numpy as np
import openmc
import openmc.deplete

# ============================================================
# USER-EDITABLE PARAMETERS
# ============================================================

# --- Geometry (cm) — from NRC AP1000 DCD ---
fuel_or      = 0.409575     # fuel pellet outer radius (0.81915 cm diameter)
gap_or       = 0.41785      # He gap outer radius = cladding inner radius
                            #   clad IR = clad OR - thickness = 0.4750 - 0.05715
clad_or      = 0.4750       # cladding outer radius  (0.950 cm diameter)
pin_pitch    = 1.26         # square lattice pitch

# --- Fuel (UO₂, sintered pellets) ---
# Use 4.8 wt% U-235 for an equilibrium-cycle pin targeting 60 GWd/MTU.
# First cycle uses 2.35/3.40/4.45 wt%; switch enrichment as needed.
enrich_u235  = 0.040        # U-235 enrichment (weight fraction in U)
uo2_td       = 10.97        # UO₂ theoretical density [g/cm³]
uo2_frac_td  = 0.95         # fraction of theoretical density
fuel_density = uo2_td * uo2_frac_td   # 10.477 g/cm³
fuel_temp    = 900.0        # fuel volume-average temperature [K]
                            #   (centreline ~1400 K, surface ~700 K)

# --- Gap (helium fill gas) ---
he_density   = 0.0015       # g/cm³  (He at ~2.5 MPa, ~700 K representative)
he_temp      = 700.0        # K

# --- Cladding: ZIRLO (Zr-1.0Nb-1.0Sn-0.1Fe, approx.) ---
clad_density = 6.56         # g/cm³
clad_temp    = 620.0        # K  (outer clad surface ~600 K)

# --- Coolant (pressurised light water, 15.5 MPa) ---
h2o_temp     = 580.0        # K  (core-average ~303 °C)
h2o_density  = 0.714        # g/cm³ at 15.5 MPa, ~303 °C

# --- Reactor power & depletion schedule ---
n_assemblies    = 157
rods_per_assy   = 264
core_power      = 3400.0e6                          # total core thermal power [W]
total_rods      = n_assemblies * rods_per_assy       # 41,448
pin_power       = core_power / total_rods            # W per rod ≈ 82 kW
pin_length      = 426.72                             # active fuel length [cm]

# Heavy-metal mass per pin (for burnup reference)
#   UO₂ volume = π r² L
#   M_UO2 = ρ × V
#   M_U   = M_UO2 × (238.03 / 270.03)
_fuel_vol_cm3   = np.pi * fuel_or**2 * pin_length
_m_uo2_g        = fuel_density * _fuel_vol_cm3
_m_u_kg         = _m_uo2_g * (238.03 / 270.03) / 1000.0

# Time [days] to reach target burnup:
#   BU = P × t / M_HM   →  t = BU × M_HM / P
target_bu_MWd_per_tU = 60_000.0
_total_days = (target_bu_MWd_per_tU * (_m_u_kg / 1000.0)
               / (pin_power / 1.0e6))

# Depletion time steps (days).
# Short steps early (Xe/Sm equilibrium, rapid reactivity swing), then lengthen.
days_per_year = 365.25

depletion_days = (
    [15.0]  * 2  +     #    30 d  (Xe-135 / Sm-149 equilibrium)
    [365.25] * 3       # 1095 d  (coarse annual steps)
)
# Append a final step to hit the target burnup exactly.
_scheduled = sum(depletion_days)
_remainder = _total_days - _scheduled
if _remainder > 0:
    depletion_days.append(_remainder)
# Now sum(depletion_days) ≈ _total_days ≈ 60 GWd/MTU

# --- Monte Carlo settings ---
batches      = 120
inactive     =  20
particles    = int(1e4)       # increase for production runs (≥ 50,000)

# --- Depletion chain ---
# Point this to a THERMAL depletion chain (PWR spectrum).
chain_file = "/home/patri/openmc/data/chain_endfb81_thermal.xml"

# --- Output directories ---
OPENMC_DIR   = "uo2_openmc"   # XMLs, H5s, statepoints, tallies
RESULTS_DIR  = "./"           # figures, CSVs

# --- Output file ---
DEPLETE_FILE = os.path.join(OPENMC_DIR, "depletion_results.h5")


# ============================================================
# FUNCTION 1: BUILD MODEL AND WRITE model.xml
# ============================================================

def build_model():
    """Construct the OpenMC model for an AP1000 pin cell.

    Returns
    -------
    model : openmc.Model
        The fully-defined model object.
    fuel_mat : openmc.Material
        Reference to the depletable fuel material (needed for post-processing).
    """

    # --- Fuel: UO₂ ---
    fuel = openmc.Material(material_id=100, name="UO2 fuel")
    fuel.set_density("g/cm3", fuel_density)
    fuel.temperature = fuel_temp

    # UO₂ stoichiometry: 1 U atom + 2 O atoms
    # Enrichment is by weight of the uranium component.
    fuel.add_nuclide("U235", enrich_u235,         percent_type="wo")
    fuel.add_nuclide("U238", 1.0 - enrich_u235,   percent_type="wo")
    fuel.add_element("O",    2.0,                  percent_type="ao")
    # OpenMC normalises ao/wo mix correctly when both types are present
    # in a single add_nuclide/add_element block. However the cleaner
    # approach for UO₂ is to use add_nuclide for U isotopes and set
    # the O/U ratio explicitly.  We rely on the helper below instead.

    # --- (cleaner UO₂ definition using built-in helper) ---
    fuel = openmc.Material(material_id=100, name="UO2 fuel")
    fuel.set_density("g/cm3", fuel_density)
    fuel.temperature = fuel_temp
    fuel.add_nuclide("U235", enrich_u235, percent_type="wo")
    fuel.add_nuclide("U238", 1.0 - enrich_u235, percent_type="wo")
    # Oxygen at correct stoichiometric ratio: 2 O per 1 U
    # Mass fraction of O in UO₂ = 2×16.00 / (238.03 + 2×16.00) ≈ 0.1185
    wt_o_in_uo2 = 2.0 * 15.999 / (238.03 + 2.0 * 15.999)
    wt_u_in_uo2 = 1.0 - wt_o_in_uo2
    # We already added U by wo, so we re-do cleanly:
    fuel = openmc.Material(material_id=100, name="UO2 fuel")
    fuel.set_density("g/cm3", fuel_density)
    fuel.temperature = fuel_temp
    # Weight fractions in UO₂
    fuel.add_nuclide("U235", wt_u_in_uo2 * enrich_u235,       percent_type="wo")
    fuel.add_nuclide("U238", wt_u_in_uo2 * (1.0 - enrich_u235), percent_type="wo")
    fuel.add_element("O",    wt_o_in_uo2,                      percent_type="wo")

    fuel.depletable = True
    fuel.volume = np.pi * fuel_or**2 * pin_length   # cm³ for one pin

    # --- Helium fill gas (fuel-clad gap) ---
    helium = openmc.Material(material_id=200, name="He gap")
    helium.set_density("g/cm3", he_density)
    helium.temperature = he_temp
    helium.add_element("He", 1.0)

    # --- Cladding: ZIRLO (Zr-1.0Nb-1.0Sn-0.1Fe approx.) ---
    clad = openmc.Material(material_id=300, name="ZIRLO cladding")
    clad.set_density("g/cm3", clad_density)
    clad.temperature = clad_temp
    clad.add_element("Zr", 0.979, percent_type="wo")
    clad.add_element("Nb", 0.010, percent_type="wo")
    clad.add_element("Sn", 0.010, percent_type="wo")
    clad.add_element("Fe", 0.001, percent_type="wo")

    # --- Coolant: light water ---
    coolant = openmc.Material(material_id=400, name="Light water")
    coolant.set_density("g/cm3", h2o_density)
    coolant.temperature = h2o_temp
    coolant.add_element("H", 2.0, percent_type="ao")
    coolant.add_element("O", 1.0, percent_type="ao")
    coolant.add_s_alpha_beta("c_H_in_H2O")

    materials = openmc.Materials([fuel, helium, clad, coolant])

    # ---- Geometry — square pin-cell with reflective boundaries ----

    fuel_surface = openmc.ZCylinder(surface_id=1, r=fuel_or)
    gap_surface  = openmc.ZCylinder(surface_id=2, r=gap_or)
    clad_surface = openmc.ZCylinder(surface_id=3, r=clad_or)

    min_z = openmc.ZPlane(surface_id=4, z0=-pin_length / 2, boundary_type="reflective")
    max_z = openmc.ZPlane(surface_id=5, z0= pin_length / 2, boundary_type="reflective")

    # Square boundary for PWR lattice (pitch × pitch box)
    left   = openmc.XPlane(surface_id=6, x0=-pin_pitch / 2, boundary_type="reflective")
    right  = openmc.XPlane(surface_id=7, x0= pin_pitch / 2, boundary_type="reflective")
    bottom = openmc.YPlane(surface_id=8, y0=-pin_pitch / 2, boundary_type="reflective")
    top    = openmc.YPlane(surface_id=9, y0= pin_pitch / 2, boundary_type="reflective")

    in_box = +left & -right & +bottom & -top & +min_z & -max_z

    fuel_cell    = openmc.Cell(cell_id=10, name="fuel",    fill=fuel,
                               region=-fuel_surface & +min_z & -max_z)
    gap_cell     = openmc.Cell(cell_id=20, name="gap",     fill=helium,
                               region=+fuel_surface & -gap_surface & +min_z & -max_z)
    clad_cell    = openmc.Cell(cell_id=30, name="clad",    fill=clad,
                               region=+gap_surface & -clad_surface & +min_z & -max_z)
    coolant_cell = openmc.Cell(cell_id=40, name="coolant", fill=coolant,
                               region=+clad_surface & in_box)

    root_universe = openmc.Universe(universe_id=1,
                                    cells=[fuel_cell, gap_cell, clad_cell, coolant_cell])
    geometry = openmc.Geometry(root_universe)

    # ---- Settings ----

    settings = openmc.Settings()
    settings.batches   = batches
    settings.inactive  = inactive
    settings.particles = particles
    settings.temperature = {"method": "interpolation"}

    r_src     = openmc.stats.Uniform(0.0, fuel_or)
    theta_src = openmc.stats.Uniform(0.0, 2 * np.pi)
    z_src     = openmc.stats.Uniform(-pin_length / 2, pin_length / 2)
    src_spatial = openmc.stats.CylindricalIndependent(r_src, theta_src, z_src)
    settings.source = openmc.IndependentSource(space=src_spatial)

    # ---- Assemble model ----

    model = openmc.Model(geometry, materials, settings)

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
    print("AP1000 Pin-Cell Depletion Model (NRC DCD)")
    print("=" * 60)
    print(f"  Fuel          : UO₂, {enrich_u235*100:.1f} wt% U-235")
    print(f"  Pellet OD     : {fuel_or*20:.3f} mm")
    print(f"  Clad OD / t   : {clad_or*20:.3f} mm / {(clad_or - gap_or)*10:.3f} mm")
    print(f"  Fuel density  : {fuel_density:.3f} g/cm³ ({uo2_frac_td*100:.1f}% TD)")
    print(f"  Cladding      : ZIRLO")
    print(f"  Coolant       : H₂O, 15.5 MPa, ~303 °C")
    print(f"  Pin power     : {pin_power:.1f} W  ({pin_power/pin_length:.1f} W/cm)")
    print(f"  HM mass/pin   : {_m_u_kg*1000:.1f} g U")
    print(f"  Target burnup : {target_bu_MWd_per_tU/1000:.0f} GWd/MTU")
    print(f"  Irrad. time   : {_total_days:.0f} d  ({_total_days/days_per_year:.2f} yr)")
    print(f"  Depletion     : {sum(depletion_days)/days_per_year:.2f} years "
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
        "O16", "O17", "O18",
        "U235", "U236", "U238",
        "Pu239", "Pu240", "Pu241", "Pu242",
        "Np237", "Am241",
        "Cs137", "Sr90", "Xe135", "Sm149",
    ]

    nuclides_tracked = [
        # Fuel chemistry
        "O16", "O17", "O18",
        # Actinides
        "U234", "U235", "U236", "U238",
        "Np237",
        "Pu238", "Pu239", "Pu240", "Pu241", "Pu242",
        "Am241", "Am242_m1", "Am243",
        "Cm242", "Cm243", "Cm244", "Cm245",
        # High-absorption fission products
        "Sm149", "Sm151", "Gd155", "Gd157",
        "Rh103", "Nd143", "Nd145", "Cs133",
        "Eu151", "Eu153", "Ag109",
        # High-inventory fission products
        "Mo95", "Mo97", "Mo98", "Mo100",
        "Xe126", "Xe128", "Xe130", "Xe131", "Xe132", "Xe134", "Xe135", "Xe136",
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

    # --- Tracked nuclides for downstream use ---
    print("\n  Tracked nuclides at EOL (atoms):")
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

    print("\n  Saving EOL inventory to CSV file...")
    import csv
    csv_path = os.path.join(RESULTS_DIR, "uo2_eol_inventory.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["nuclide", "atoms"])
        for nuc, val in inventory:
            if val > 0:
                writer.writerow([nuc, f"{val:.6e}"])
    print(f"  Saved to {csv_path}")

    # --- Save tracked nuclides as separate CSV ---
    print("\n  Saving tracked nuclide inventory to CSV file...")
    csv_tracked_path = os.path.join(RESULTS_DIR, "uo2_eol_inventory_tracked.csv")
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
        ax.set_title(r"AP1000 Pin-Cell Depletion — UO$_2$ / He / ZIRLO / H$_2$O")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, "uo2_keff_vs_time.png")
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