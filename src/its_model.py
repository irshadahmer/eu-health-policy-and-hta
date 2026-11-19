from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


@dataclass
class ITSResult:
    country: str
    outcome: str
    intervention_year: int
    pre_trend_slope: float
    level_change: float
    level_change_pval: float
    slope_change: float
    slope_change_pval: float
    counterfactual_2023: float
    observed_2023: float
    df_fitted: pd.DataFrame


def fit_its_segmented_regression(
    panel: pd.DataFrame,
    iso2: str = "HU",
    outcome: str = "preventable_mortality_sdr",
    intervention_year: int = 2020,
) -> ITSResult:
    sub = panel[panel["iso2"] == iso2].dropna(subset=[outcome]).sort_values("year").copy()
    if len(sub) < 6:
        raise ValueError(f"Insufficient annual observations for {iso2} on {outcome}")

    # Standard ITS parameterization:
    # time: running year index starting at 0
    # step: 1 from intervention_year onward (immediate level shift)
    # time_after: years elapsed since intervention_year (sustained slope change)
    sub["time"] = sub["year"] - sub["year"].min()
    sub["step"] = (sub["year"] >= intervention_year).astype(int)
    sub["time_after"] = np.where(sub["year"] >= intervention_year, sub["year"] - intervention_year, 0)

    # Fit OLS with Newey-West HAC covariance (maxlags=1) for serial correlation in annual rates
    formula = f"{outcome} ~ time + step + time_after"
    model = smf.ols(formula, data=sub).fit(cov_type="HAC", cov_kwds={"maxlags": 1})

    sub["fitted"] = model.fittedvalues

    # Counterfactual trajectory assuming pre-intervention trend continued uninterrupted
    cf_df = sub.copy()
    cf_df["step"] = 0
    cf_df["time_after"] = 0
    sub["counterfactual"] = model.predict(cf_df)

    row_2023 = sub[sub["year"] == 2023]
    obs_23 = float(row_2023[outcome].iloc[0]) if not row_2023.empty else float("nan")
    cf_23 = float(row_2023["counterfactual"].iloc[0]) if not row_2023.empty else float("nan")

    return ITSResult(
        country=sub["country"].iloc[0],
        outcome=outcome,
        intervention_year=intervention_year,
        pre_trend_slope=float(model.params["time"]),
        level_change=float(model.params["step"]),
        level_change_pval=float(model.pvalues["step"]),
        slope_change=float(model.params["time_after"]),
        slope_change_pval=float(model.pvalues["time_after"]),
        counterfactual_2023=cf_23,
        observed_2023=obs_23,
        df_fitted=sub,
    )


def main() -> None:
    panel = pd.read_parquet(PROCESSED_DIR / "eu_health_panel.parquet")
    for code in ["HU", "ES", "FR", "PL"]:
        res = fit_its_segmented_regression(panel, iso2=code, outcome="preventable_mortality_sdr", intervention_year=2020)
        print(
            f"{res.country:<10} | Pre-slope: {res.pre_trend_slope:+.2f}/yr | "
            f"2020 Shock (step): {res.level_change:+.2f} (p={res.level_change_pval:.3f}) | "
            f"Post-slope delta: {res.slope_change:+.2f}/yr (p={res.slope_change_pval:.3f}) | "
            f"2023 Obs vs CF: {res.observed_2023:.1f} vs {res.counterfactual_2023:.1f}"
        )


if __name__ == "__main__":
    main()
