from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

EU27_NAMES = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CY": "Cyprus",
    "CZ": "Czechia", "DE": "Germany", "DK": "Denmark", "EE": "Estonia",
    "EL": "Greece", "ES": "Spain", "FI": "Finland", "FR": "France",
    "HR": "Croatia", "HU": "Hungary", "IE": "Ireland", "IT": "Italy",
    "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia", "MT": "Malta",
    "NL": "Netherlands", "PL": "Poland", "PT": "Portugal", "RO": "Romania",
    "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia",
}

EUPUBHEALTH_CONSORTIUM = {"FR", "ES", "PL", "HU", "IE", "NL", "DK"}


def _melt_years(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    geo_col = [c for c in df.columns if c.startswith("geo")][0]
    year_cols = [c for c in df.columns if c.isdigit()]
    out = df.melt(
        id_vars=[geo_col],
        value_vars=year_cols,
        var_name="year",
        value_name=value_name,
    )
    out = out.rename(columns={geo_col: "iso2"})
    out["year"] = out["year"].astype(int)
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    return out[out["iso2"].isin(EU27_NAMES)].copy()


def build_eu_health_panel() -> pd.DataFrame:
    mort = pd.read_parquet(RAW_DIR / "treatable_mortality_hlth_cd_apr.parquet")
    exp = pd.read_parquet(RAW_DIR / "health_expenditure_hlth_sha11_hf.parquet")
    le = pd.read_parquet(RAW_DIR / "life_expectancy_demo_mlexpec.parquet")

    # Standardized death rate (RT) per 100k for total population, all ICD-10 causes
    trt = mort[(mort["mortalit"] == "TRT") & (mort["sex"] == "T") & (mort["icd10"] == "TOTAL") & (mort["unit"] == "RT")]
    prvt = mort[(mort["mortalit"] == "PRVT") & (mort["sex"] == "T") & (mort["icd10"] == "TOTAL") & (mort["unit"] == "RT")]

    df_trt = _melt_years(trt, "treatable_mortality_sdr")
    df_prvt = _melt_years(prvt, "preventable_mortality_sdr")

    # Total healthcare expenditure (TOT_HF): PPS per capita & % of GDP
    exp_pps = exp[(exp["icha11_hf"] == "TOT_HF") & (exp["unit"] == "PPS_HAB")]
    exp_gdp = exp[(exp["icha11_hf"] == "TOT_HF") & (exp["unit"] == "PC_GDP")]

    df_pps = _melt_years(exp_pps, "health_exp_pps_hab")
    df_gdp = _melt_years(exp_gdp, "health_exp_pct_gdp")

    # Life expectancy at birth (Y_LT1 in Eurostat demo_mlexpec)
    birth_code = "Y_LT1" if "Y_LT1" in le["age"].values else "Y1"
    le_birth = le[(le["age"] == birth_code) & (le["sex"] == "T") & (le["unit"] == "YR")]
    df_le = _melt_years(le_birth, "life_expectancy_yrs")

    panel = (
        df_trt.merge(df_prvt, on=["iso2", "year"], how="outer")
        .merge(df_pps, on=["iso2", "year"], how="outer")
        .merge(df_gdp, on=["iso2", "year"], how="outer")
        .merge(df_le, on=["iso2", "year"], how="outer")
    )

    panel = panel[(panel["year"] >= 2011) & (panel["year"] <= 2023)].copy()
    panel["country"] = panel["iso2"].map(EU27_NAMES)
    panel["is_consortium"] = panel["iso2"].isin(EUPUBHEALTH_CONSORTIUM)
    panel["avoidable_mortality_sdr"] = panel["treatable_mortality_sdr"] + panel["preventable_mortality_sdr"]

    return panel.sort_values(["iso2", "year"]).reset_index(drop=True)


def estimate_efficiency_frontier(panel: pd.DataFrame, target_year: int = 2022) -> pd.DataFrame:
    sub = panel[panel["year"] == target_year].dropna(
        subset=["treatable_mortality_sdr", "health_exp_pps_hab"]
    ).copy()

    # Log-linear Preston-style specification: diminishing mortality reductions per PPS spent
    X = sm.add_constant(np.log(sub["health_exp_pps_hab"]))
    model = sm.OLS(sub["treatable_mortality_sdr"], X).fit()

    sub["expected_treatable_sdr"] = model.predict(X)
    # Negative residual = lower mortality than spending predicts (higher technical efficiency)
    sub["mortality_excess_vs_frontier"] = sub["treatable_mortality_sdr"] - sub["expected_treatable_sdr"]
    return sub.sort_values("mortality_excess_vs_frontier").reset_index(drop=True)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_eu_health_panel()
    out_file = PROCESSED_DIR / "eu_health_panel.parquet"
    panel.to_parquet(out_file, index=False)

    frontier = estimate_efficiency_frontier(panel, target_year=2022)
    print(f"Saved clean panel: {len(panel)} country-years -> {out_file.name}")
    print("\n2022 Europubhealth+ Consortium Snapshot:")
    cols = ["country", "health_exp_pps_hab", "treatable_mortality_sdr", "mortality_excess_vs_frontier", "life_expectancy_yrs"]
    print(frontier[frontier["is_consortium"]][cols].to_string(index=False))


if __name__ == "__main__":
    main()
