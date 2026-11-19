import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.disparities import estimate_efficiency_frontier
from src.its_model import fit_its_segmented_regression
from src.markov_hta import HTAParams, run_probabilistic_sensitivity

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FIG_DIR = ROOT_DIR / "outputs" / "figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})


def plot_efficiency_frontier(panel: pd.DataFrame) -> None:
    df = estimate_efficiency_frontier(panel, target_year=2022)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    non_cons = df[~df["is_consortium"]]
    cons = df[df["is_consortium"]]

    ax.scatter(
        non_cons["health_exp_pps_hab"],
        non_cons["treatable_mortality_sdr"],
        color="#94a3b8",
        alpha=0.75,
        s=55,
        label="Other EU-27 Member States",
    )
    ax.scatter(
        cons["health_exp_pps_hab"],
        cons["treatable_mortality_sdr"],
        color="#0f766e",
        edgecolor="#042f2e",
        linewidth=1.1,
        s=95,
        zorder=4,
        label="Europubhealth+ Consortium Countries",
    )

    x_grid = np.linspace(df["health_exp_pps_hab"].min() * 0.95, df["health_exp_pps_hab"].max() * 1.03, 200)
    coeffs = np.polyfit(np.log(df["health_exp_pps_hab"]), df["treatable_mortality_sdr"], 1)
    y_grid = coeffs[0] * np.log(x_grid) + coeffs[1]
    ax.plot(x_grid, y_grid, color="#dc2626", linestyle="--", linewidth=1.8, label="EU Log-Linear Efficiency Frontier")

    for _, row in cons.iterrows():
        offset_y = 6 if row["iso2"] not in {"NL", "FR"} else -11
        ax.annotate(
            row["country"],
            (row["health_exp_pps_hab"], row["treatable_mortality_sdr"]),
            xytext=(5, offset_y),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color="#1e293b",
        )

    ax.set_title("EU-27 Health System Efficiency: Spending vs. Treatable Mortality (2022)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Current Healthcare Expenditure (PPS per Inhabitant)", fontsize=10)
    ax.set_ylabel("Treatable Mortality (Standardized Death Rate / 100k)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "eu_efficiency_frontier.png")
    plt.close(fig)


def plot_its_comparison(panel: pd.DataFrame) -> None:
    countries = ["HU", "PL", "ES", "FR"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    axes = axes.flatten()

    for ax, code in zip(axes, countries):
        res = fit_its_segmented_regression(panel, iso2=code, outcome="preventable_mortality_sdr", intervention_year=2020)
        df = res.df_fitted

        ax.scatter(df["year"], df["preventable_mortality_sdr"], color="#1e293b", s=28, zorder=4, label="Observed SDR")
        ax.plot(df["year"], df["fitted"], color="#0284c7", linewidth=2, label="ITS Segmented Fit")

        post = df[df["year"] >= 2020]
        ax.plot(post["year"], post["counterfactual"], color="#dc2626", linestyle="--", linewidth=1.6, label="Pre-2020 Counterfactual")
        ax.axvline(2019.5, color="#64748b", linestyle=":", linewidth=1.2)

        ax.set_title(
            f"{res.country} (2020 Shock: {res.level_change:+.1f}, p={res.level_change_pval:.3f})",
            fontsize=10,
            fontweight="bold",
        )
        ax.grid(True, linestyle=":", alpha=0.4)

    axes[0].set_ylabel("Preventable Mortality SDR / 100k", fontsize=9.5)
    axes[2].set_ylabel("Preventable Mortality SDR / 100k", fontsize=9.5)
    axes[0].legend(loc="upper left", fontsize=7.5, frameon=True)

    fig.suptitle("Interrupted Time-Series: Preventable Mortality Structural Break (2011-2023)", fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "its_policy_shock.png")
    plt.close(fig)


def plot_hta_psa_plane() -> None:
    params = HTAParams()
    psa = run_probabilistic_sensitivity(params, n_sim=1000, seed=42)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(psa["inc_qaly"], psa["inc_cost"], alpha=0.45, s=22, color="#2563eb", label="Monte Carlo Iterations (n=1,000)")

    ax.scatter(
        [psa["inc_qaly"].mean()],
        [psa["inc_cost"].mean()],
        color="#dc2626",
        s=90,
        marker="D",
        zorder=5,
        label=f"Mean Incremental (dQALY={psa['inc_qaly'].mean():+.3f}, dCost=EUR {psa['inc_cost'].mean():+.0f})",
    )

    q_range = np.linspace(min(0, psa["inc_qaly"].min() * 1.1), psa["inc_qaly"].max() * 1.1, 100)
    ax.plot(q_range, q_range * params.wtp_threshold_eur, color="#059669", linestyle="--", linewidth=1.6, label="WTP Threshold (EUR 30,000 / QALY)")

    ax.axhline(0, color="#475569", linewidth=0.9)
    ax.axvline(0, color="#475569", linewidth=0.9)

    ax.set_title("Probabilistic Sensitivity Analysis: Rapid Point-of-Care Diagnostic Triage", fontsize=11.5, fontweight="bold", pad=12)
    ax.set_xlabel("Incremental QALYs (10-Year Discounted Horizon)", fontsize=10)
    ax.set_ylabel("Incremental Costs (EUR per Patient)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "hta_psa_plane.png")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PROCESSED_DIR / "eu_health_panel.parquet")
    plot_efficiency_frontier(panel)
    plot_its_comparison(panel)
    plot_hta_psa_plane()
    print("Generated 3 PNG figures in outputs/figures/:")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
