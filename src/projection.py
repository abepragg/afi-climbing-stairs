"""
Synthetic Household Micro-simulation for AFI 'climbing stairs' project.

Inputs (CSV in data/):
  - prep_PEC19.csv           : Year, Sex, AgeNum, VALUE(or Population), (optional) Criteria for Projection
  - baseline.csv             : Band (0-3/4-64/65+), Sex (Male/Female), Baseline % (decimal) or baseline_rate
  - Households_size_2022.csv : size_label, size_numeric (1..7; 7 for '7+'), households_2022

Outputs (written to outputs/):
  - microsim_summary.csv           : mean & 95% CI for pA, pB, A∪B by year
  - microsim_bars.png, microsim_lines.png : publication-ready charts

Model:
  1) Fix 2022 person-level rates by Band×Sex (from baseline) for “stairs difficulty”.
  2) For each target year, draw a large synthetic household sample using the 2022 size mix.
  3) Fill each household with members sampled i.i.d. from the year’s Age×Sex population margins.
  4) For each person, assign:
      - child_0_3 = (AgeNum in 0..3)
      - stairs_difficulty ~ Bernoulli(rate[Band(AgeNum) × Sex])
  5) For each household, compute:
      - A = any member has stairs_difficulty
      - B = any member is child_0_3
      - A∪B
  6) Repeat (Monte Carlo) to get uncertainty bands.

Assumptions:
  - Household-size distribution stays at the 2022 mix across projection years (documented).
  - Within-household composition is random given Age×Sex margins (no extra structure available).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------
# Paths & parameters
# -------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / "data"
OUT_DIR   = REPO_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# File names (adjust if your CSV names differ)
POP_CSV   = DATA_DIR / "prep_PEC19.csv"
BASE_CSV  = DATA_DIR / "baseline.csv"
HHS_CSV   = DATA_DIR / "Households_size_2022.csv"

# Simulation controls
N_HOUSEHOLDS   = 120_000     # synthetic households per run (trade-off: precision vs speed)
N_REPS         = 300         # Monte Carlo repetitions per year (>=200 gives stable CIs)
RANDOM_SEED    = 42          # reproducible runs
YEARS_FOCUS    = [2022, 2030, 2040, 2050]  # subset of years to summarise/plot (auto-filtered to existing)


# -------------------
# Utilities
# -------------------
def read_csv_norm(path: Path) -> pd.DataFrame:
    """Read CSV and normalise column names to lower-case stripped strings."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def band_from_age(age: np.ndarray) -> np.ndarray:
    """Map single ages to string bands '0-3', '4-64', '65+' (vectorised)."""
    out = np.empty(age.shape, dtype=object)
    out[age <= 3] = "0-3"
    out[(age >= 4) & (age <= 64)] = "4-64"
    out[age >= 65] = "65+"
    return out


def summarise_ci(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (mean, lower, upper) where lower/upper are (100*(alpha/2))% and (100*(1-alpha/2))% quantiles."""
    mean = float(np.mean(x))
    lo   = float(np.quantile(x, alpha/2))
    hi   = float(np.quantile(x, 1 - alpha/2))
    return mean, lo, hi


# -------------------
# Load & prepare data
# -------------------
pop = read_csv_norm(POP_CSV)
base = read_csv_norm(BASE_CSV)
hhs  = read_csv_norm(HHS_CSV)

# Population: accept 'value' or 'population'
if "population" not in pop.columns:
    if "value" in pop.columns:
        pop = pop.rename(columns={"value": "population"})
    else:
        raise ValueError("prep_PEC19.csv must have a 'population' or 'VALUE' column.")

# Keep only columns we rely on
required_pop = {"year", "sex", "agenum", "population"}
missing = required_pop - set(pop.columns)
if missing:
    raise ValueError(f"prep_PEC19.csv missing columns: {missing}")

# Filter to Male/Female only (drop 'Both sexes' if any) and standardise title case
pop = pop[pop["sex"].str.lower().isin(["male", "female"])].copy()
pop["sex"] = pop["sex"].str.title()

# Baseline rates: accept 'baseline %' or 'baseline_rate'
if "baseline_rate" not in base.columns:
    if "baseline %" in base.columns:
        base = base.rename(columns={"baseline %": "baseline_rate"})
    else:
        raise ValueError("baseline.csv must have 'Baseline %' (decimal) or 'baseline_rate' column.")
# Ensure columns exist
required_base = {"band", "sex", "baseline_rate"}
if not required_base.issubset(set(base.columns)):
    raise ValueError("baseline.csv must have columns: Band, Sex, baseline_rate (decimal).")
base["sex"] = base["sex"].str.title()
base["band"] = base["band"].str.strip()

# Household size: compute shares
required_hhs = {"size_numeric", "households_2022"}
if not required_hhs.issubset(set(hhs.columns)):
    raise ValueError("Households_size_2022.csv must have size_numeric and households_2022.")
hhs["size_numeric"] = pd.to_numeric(hhs["size_numeric"], errors="coerce").fillna(7).astype(int)
hhs = hhs.sort_values("size_numeric")
hhs["share_hs"] = hhs["households_2022"] / hhs["households_2022"].sum()

# -------------------
# Build year-specific samplers
# -------------------
def make_person_sampler(pop_year: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    From a Year slice of pop with columns (sex, agenum, population), build:
      - ages array (int), sexes array (str), probabilities array (float), baseline_rate array aligned to entries.
    Sampling is over (Age, Sex) categories with prob proportional to population.
    """
    # Aggregate in case input has duplicates
    g = pop_year.groupby(["sex", "agenum"], as_index=False)["population"].sum()
    g = g.sort_values(["sex", "agenum"])
    probs = g["population"] / g["population"].sum()

    ages = g["agenum"].to_numpy()
    sexes = g["sex"].to_numpy()

    # Map each (age, sex) to a baseline_rate via band mapping
    bands = band_from_age(ages)
    # Merge baseline rates for vectorised lookup
    base_map = base.set_index(["band", "sex"])["baseline_rate"]
    # Create rate array by iterating (fast enough for age range)
    rates = np.array([base_map.get((bands[i], sexes[i]), np.nan) for i in range(len(ages))], dtype=float)
    if np.isnan(rates).any():
        raise ValueError("Missing baseline_rate for some Band×Sex. Check baseline.csv values.")

    return ages, sexes, probs.to_numpy(), rates


# -------------------
# One microsim run for a single year
# -------------------
def run_year_once(year: int, n_households: int, rng: np.random.Generator) -> dict:
    """
    Simulate one synthetic sample for a given year and return household-level shares:
      pA (stairs), pB (child 0-3), pUnion (A∪B).
    """
    # Year-specific population margins & sampler
    pop_y = pop[pop["year"] == year]
    if pop_y.empty:
        raise ValueError(f"No population rows for year {year}.")
    ages, sexes, probs, rates = make_person_sampler(pop_y)

    # Draw household sizes according to 2022 mix
    sizes = hhs["size_numeric"].to_numpy()
    size_p = hhs["share_hs"].to_numpy()
    hh_sizes = rng.choice(sizes, size=n_households, p=size_p)

    # Total people to simulate
    n_people = int(hh_sizes.sum())

    # Sample (Age, Sex) categories for each person
    # idx draws from categories; we then map to ages/sexes/rates arrays
    idx = rng.choice(len(ages), size=n_people, p=probs)
    age_draws = ages[idx]
    sex_draws = sexes[idx]
    rate_draws = rates[idx]

    # Person-level indicators
    has_child_0_3 = (age_draws <= 3)                      # deterministic for B
    has_stairs    = rng.random(n_people) < rate_draws     # Bernoulli for A

    # Collapse persons to households: compute "any" within each household slice
    # Build start indices for reduceat
    boundaries = np.cumsum(hh_sizes)
    starts = np.concatenate(([0], boundaries[:-1]))

    # For boolean arrays, we use max reduceat to get 'any'
    # Convert to uint8 (0/1), then take max per segment
    a_uint = has_stairs.astype(np.uint8)
    b_uint = has_child_0_3.astype(np.uint8)

    # Sum per household and threshold >0 (faster than reduceat(max))
    sumA = np.add.reduceat(a_uint, starts)
    sumB = np.add.reduceat(b_uint, starts)
    A = sumA > 0
    B = sumB > 0

    # Shares
    pA = A.mean()
    pB = B.mean()
    pUnion = (A | B).mean()

    return {"year": year, "pA": pA, "pB": pB, "pUnion": pUnion}


def run_year_many(year: int, n_households: int, n_reps: int, seed: int) -> dict:
    """Repeat the simulation n_reps times and return mean & 95% CI for pA, pB, and A∪B."""
    rng = np.random.default_rng(seed)
    pA_list, pB_list, pU_list = [], [], []
    for r in range(n_reps):
        res = run_year_once(year, n_households, rng)
        pA_list.append(res["pA"])
        pB_list.append(res["pB"])
        pU_list.append(res["pUnion"])
    pA_mean, pA_lo, pA_hi = summarise_ci(np.array(pA_list))
    pB_mean, pB_lo, pB_hi = summarise_ci(np.array(pB_list))
    pU_mean, pU_lo, pU_hi = summarise_ci(np.array(pU_list))
    return {
        "year": year,
        "pA_mean": pA_mean, "pA_lo": pA_lo, "pA_hi": pA_hi,
        "pB_mean": pB_mean, "pB_lo": pB_lo, "pB_hi": pB_hi,
        "pUnion_mean": pU_mean, "pUnion_lo": pU_lo, "pUnion_hi": pU_hi,
        "n_households": n_households, "n_reps": n_reps
    }


# -------------------
# Driver
# -------------------
def main():
    years_available = sorted(pop["year"].unique().tolist())
    years = [y for y in YEARS_FOCUS if y in years_available]
    if not years:
        years = years_available  # fall back to all years present

    results = []
    for y in years:
        stats = run_year_many(y, N_HOUSEHOLDS, N_REPS, RANDOM_SEED + y)
        results.append(stats)

    out = pd.DataFrame(results)
    out_path = OUT_DIR / "microsim_summary.csv"
    out.to_csv(out_path, index=False)

    # ---------- Charts ----------
    # Bars (with 95% CI error bars)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(out))
    width = 0.28

    def add_bars(center, mean, lo, hi, label):
        ax.bar(center, 100*mean, width=width, label=label)
        # error bars as vertical lines
        for xi, m, l, h in zip(center, mean, lo, hi):
            ax.plot([xi, xi], [100*l, 100*h], linewidth=2)

    add_bars(x - width, out["pA_mean"], out["pA_lo"], out["pA_hi"], "Households: stairs (A)")
    add_bars(x,           out["pB_mean"], out["pB_lo"], out["pB_hi"], "Households: 0–3 child (B)")
    add_bars(x + width,   out["pUnion_mean"], out["pUnion_lo"], out["pUnion_hi"], "A ∪ B")

    ax.set_xticks(x)
    ax.set_xticklabels(out["year"].astype(str).tolist())
    ax.set_ylabel("Share of households (%)")
    ax.set_title("Microsimulated household shares with 95% intervals")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "microsim_bars.png", dpi=200)
    plt.close(fig)

    # Lines (mean + shaded 95% interval)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(out["year"], 100*out["pA_mean"], marker="o", label="A: stairs")
    ax.fill_between(out["year"], 100*out["pA_lo"], 100*out["pA_hi"], alpha=0.15)

    ax.plot(out["year"], 100*out["pB_mean"], marker="s", label="B: 0–3 child")
    ax.fill_between(out["year"], 100*out["pB_lo"], 100*out["pB_hi"], alpha=0.15)

    ax.plot(out["year"], 100*out["pUnion_mean"], marker="^", label="A ∪ B")
    ax.fill_between(out["year"], 100*out["pUnion_lo"], 100*out["pUnion_hi"], alpha=0.15)

    ax.set_ylabel("Share of households (%)")
    ax.set_title("Microsimulated projections (Method M2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "microsim_lines.png", dpi=200)
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"Saved charts to: {OUT_DIR/'microsim_bars.png'} and {OUT_DIR/'microsim_lines.png'}")


if __name__ == "__main__":
    main()