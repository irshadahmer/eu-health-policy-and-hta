from pathlib import Path
import eurostat
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

DATASETS = {
    "treatable_mortality": "hlth_cd_apr",      # Amenable & preventable mortality (SDR per 100k)
    "health_expenditure": "hlth_sha11_hf",     # Healthcare expenditure (% GDP & PPS per capita)
    "life_expectancy": "demo_mlexpec",         # Life expectancy at birth
}


def fetch_and_cache(code: str, out_path: Path) -> pd.DataFrame:
    df = eurostat.get_data_df(code, flags=False)
    if df is None or df.empty:
        raise RuntimeError(f"Empty response from Eurostat for dataset '{code}'")
    df.to_parquet(out_path, index=False)
    return df


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, code in DATASETS.items():
        target = RAW_DIR / f"{name}_{code}.parquet"
        print(f"Fetching {code} -> {target.name}...")
        df = fetch_and_cache(code, target)
        print(f"  Saved {len(df):,} rows ({df.shape[1]} cols)")


if __name__ == "__main__":
    main()
