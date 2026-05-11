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
enrich_u235  = 0.19        # U-235 enrichment (weight fraction), outer core
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
total_years   = 30.0

# Build a schedule: 6 steps of 60d, then 10 steps of 180d, then remainder in 365d steps
depletion_days = (
    [60.0]  * 6  +     #   360 d  (year 1)
    [180.0] * 10 +     # 1,800 d  (years 2–6)
    [365.25] * 24       # 8,766 d  (years 7–30)
)
# Total ≈ 10,926 d ≈ 29.9 yr — close enough to 30 yr

# --- Monte Carlo settings ---
batches      = 120
inactive     = 40
particles    = 10000       # increase for production runs (≥50 000)

# ============================================================
# MATERIALS
# ============================================================

# --- Fuel: U-10Zr ---
fuel = openmc.Material(name="U-10Zr fuel")
fuel.set_density("g/cm3", fuel_density)
fuel.temperature = fuel_temp

# Weight fractions of U and Zr
wt_frac_u = 1.0 - wt_frac_zr

# Add uranium isotopes (by weight within the uranium portion)
fuel.add_nuclide("U235", wt_frac_u * enrich_u235,       percent_type="wo")
fuel.add_nuclide("U238", wt_frac_u * (1.0 - enrich_u235), percent_type="wo")

# Add zirconium (natural isotopic mix)
fuel.add_element("Zr", wt_frac_zr, percent_type="wo")

# Mark as depletable
fuel.depletable = True
# Assign a volume for depletion normalisation
fuel.volume = np.pi * fuel_or**2 * pin_length   # cm³ for one pin

# --- Sodium bond (same Na, in fuel-clad gap) ---
na_bond = openmc.Material(name="Na bond")
na_bond.set_density("g/cm3", na_density)
na_bond.temperature = fuel_temp   # bond sodium near fuel temperature
na_bond.add_element("Na", 1.0)

# --- Cladding: HT-9 (simplified) ---
clad = openmc.Material(name="HT-9 cladding")
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
coolant = openmc.Material(name="Na coolant")
coolant.set_density("g/cm3", na_density)
coolant.temperature = na_temp
coolant.add_element("Na", 1.0)

materials = openmc.Materials([fuel, na_bond, clad, coolant])
materials.export_to_xml()

# ============================================================
# GEOMETRY  — hexagonal pin-cell with reflective boundaries
# ============================================================

fuel_surface = openmc.ZCylinder(r=fuel_or)
bond_surface = openmc.ZCylinder(r=bond_or)
clad_surface = openmc.ZCylinder(r=clad_or)

# Hexagonal prism bounding the unit cell (infinite in z)
hex_prism = openmc.model.HexagonalPrism(
    edge_length=pin_pitch / np.sqrt(3.0),
    orientation="y",
    boundary_type="reflective",
)

fuel_cell    = openmc.Cell(name="fuel",    fill=fuel,    region=-fuel_surface)
bond_cell    = openmc.Cell(name="bond",    fill=na_bond, region=+fuel_surface & -bond_surface)
clad_cell    = openmc.Cell(name="clad",    fill=clad,    region=+bond_surface & -clad_surface)
coolant_cell = openmc.Cell(name="coolant", fill=coolant, region=+clad_surface & -hex_prism)

root_universe = openmc.Universe(cells=[fuel_cell, bond_cell, clad_cell, coolant_cell])
geometry = openmc.Geometry(root_universe)
geometry.export_to_xml()

# ============================================================
# SETTINGS
# ============================================================

settings = openmc.Settings()
settings.batches  = batches
settings.inactive = inactive
settings.particles = particles
settings.temperature = {
    "method": "interpolation",
    "multipole": True,
}

# Initial source — uniform in the fuel slug
r_src = openmc.stats.Uniform(0.0, fuel_or)
theta_src = openmc.stats.Uniform(0.0, 2 * np.pi)
z_src = openmc.stats.Uniform(-pin_length / 2, pin_length / 2)
src_spatial = openmc.stats.CylindricalIndependent(r_src, theta_src, z_src)
settings.source = openmc.IndependentSource(space=src_spatial)

settings.export_to_xml()

# ============================================================
# TALLIES (optional — k-eff is tracked automatically)
# ============================================================

tallies = openmc.Tallies()

# Flux spectrum in the fuel (for sanity-checking the fast spectrum)
energy_filter = openmc.EnergyFilter(np.logspace(np.log10(1e-5), np.log10(20e6), 201))
cell_filter   = openmc.CellFilter(fuel_cell)

flux_tally = openmc.Tally(name="fuel flux spectrum")
flux_tally.filters = [cell_filter, energy_filter]
flux_tally.scores  = ["flux"]
tallies.append(flux_tally)

# Fission rate in fuel
fission_tally = openmc.Tally(name="fission rate")
fission_tally.filters = [cell_filter]
fission_tally.scores  = ["fission"]
tallies.append(fission_tally)

tallies.export_to_xml()

# ============================================================
# DEPLETION
# ============================================================

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
print(f"  Pin power     : {pin_power:.1f} W  ({pin_power/pin_length*100:.1f} W/cm)")
print(f"  Assembly      : {pins_per_assy} pins  (1 of {n_assemblies} in core)")
print(f"  Assy power    : {assy_power/1e6:.3f} MWth")
print(f"  Depletion     : {sum(depletion_days)/days_per_year:.1f} years "
      f"in {len(depletion_days)} steps")
print(f"  MC particles  : {particles} / batch × {batches} batches")
print("=" * 60)

# ENDF/B-VIII.0 SFR depletion chain
chain_file = "/Users/patrickpark/git-repos/openmc-data/endfb-8.0-hdf5/chain_endfb80_sfr.xml"

# Create the transport operator
model = openmc.Model(geometry, materials, settings)

operator = openmc.deplete.CoupledOperator(
    model,
    chain_file=chain_file,
    normalization_mode="source-rate",
)

# Use the predictor-corrector (CE/CM) integrator
integrator = openmc.deplete.CECMIntegrator(
    operator,
    depletion_days,                # time steps in days
    power=pin_power,               # constant power per pin [W]
    timestep_units="d",
)

# Run the depletion calculation
integrator.integrate()

print("\nDepletion calculation complete.")

# ============================================================
# POST-PROCESSING  — extract k-eff vs. time and key nuclides
# ============================================================

results = openmc.deplete.Results("depletion_results.h5")

# Extract k-effective
time_steps, keffs = results.get_keff()
time_years = time_steps / days_per_year    # convert days → years

print("\n  Time [yr]     k-eff       ± σ")
print("  " + "-" * 38)
for t, (k, sig) in zip(time_years, keffs):
    print(f"  {t:8.2f}     {k:.5f}   ± {sig:.5f}")

# Extract selected nuclide inventories (atoms) in the fuel
nuclides_of_interest = [
    "U235", "U238", "Pu239", "Pu240", "Pu241",
    "Zr93", "Cs137", "Sr90", "Xe135",
]

print("\n  Nuclide inventories at end of life (atoms):")
print("  " + "-" * 44)

_, atoms = results.get_atoms(fuel, nuclides_of_interest)
for i, nuc in enumerate(nuclides_of_interest):
    print(f"  {nuc:8s}  {atoms[i, -1]:.4e}")

# ============================================================
# OPTIONAL: save a quick matplotlib plot of k-eff vs burnup
# ============================================================

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
    fig.savefig("keff_vs_time.png", dpi=150)
    print("\n  Plot saved to keff_vs_time.png")
except ImportError:
    print("\n  matplotlib not available — skipping plot.")