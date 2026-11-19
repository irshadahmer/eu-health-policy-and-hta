from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.disparities import EU27_NAMES, estimate_efficiency_frontier
from src.its_model import fit_its_segmented_regression
from src.markov_hta import HTAParams, evaluate_base_case, run_probabilistic_sensitivity

st.set_page_config(
    page_title="EU Health Policy & HTA Analytics",
    layout="wide",
)

DATA_PATH = Path(__file__).resolve().parent / "data" / "processed" / "eu_health_panel.parquet"


@st.cache_data
def load_panel() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


def render_disparities_tab(panel: pd.DataFrame) -> None:
    st.subheader("EU-27 Health System Efficiency & Avoidable Mortality")
    st.caption("Source: Eurostat (hlth_cd_apr, hlth_sha11_hf, demo_mlexpec)")

    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    available_years = sorted(panel.dropna(subset=["treatable_mortality_sdr", "health_exp_pps_hab"])["year"].unique())
    selected_year = col_ctrl1.selectbox("Reference Year", available_years, index=len(available_years) - 1)
    highlight_consortium = col_ctrl2.checkbox("Highlight Europubhealth+ Consortium Countries (FR, ES, PL, HU, IE, NL, DK)", value=True)

    frontier_df = estimate_efficiency_frontier(panel, target_year=selected_year)
    frontier_df["Group"] = np.where(
        frontier_df["is_consortium"] & highlight_consortium,
        "Europubhealth+ Consortium",
        "EU-27 Member State",
    )

    fig = px.scatter(
        frontier_df,
        x="health_exp_pps_hab",
        y="treatable_mortality_sdr",
        color="Group",
        text="iso2",
        hover_name="country",
        hover_data={
            "health_exp_pps_hab": ":.1f",
            "treatable_mortality_sdr": ":.2f",
            "mortality_excess_vs_frontier": ":+.2f",
            "life_expectancy_yrs": ":.1f",
            "iso2": False,
        },
        color_discrete_map={
            "Europubhealth+ Consortium": "#0f766e",
            "EU-27 Member State": "#94a3b8",
        },
        labels={
            "health_exp_pps_hab": "Healthcare Expenditure (PPS per Inhabitant)",
            "treatable_mortality_sdr": "Treatable Mortality SDR (per 100,000)",
        },
        height=500,
    )
    fig.update_traces(textposition="top center", marker=dict(size=11))

    # Overlay fitted log-linear frontier
    ordered = frontier_df.sort_values("health_exp_pps_hab")
    fig.add_trace(
        go.Scatter(
            x=ordered["health_exp_pps_hab"],
            y=ordered["expected_treatable_sdr"],
            mode="lines",
            name="Log-Linear Efficiency Frontier",
            line=dict(color="#dc2626", dash="dash", width=2),
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Efficiency Residual Ranking (Negative = Outperforming Spending Prediction)")
    display_cols = [
        "country",
        "health_exp_pps_hab",
        "health_exp_pct_gdp",
        "treatable_mortality_sdr",
        "expected_treatable_sdr",
        "mortality_excess_vs_frontier",
        "life_expectancy_yrs",
    ]
    st.dataframe(
        frontier_df[display_cols].rename(
            columns={
                "country": "Country",
                "health_exp_pps_hab": "Exp (PPS/cap)",
                "health_exp_pct_gdp": "Exp (% GDP)",
                "treatable_mortality_sdr": "Treatable SDR",
                "expected_treatable_sdr": "Frontier Pred SDR",
                "mortality_excess_vs_frontier": "Excess Mortality",
                "life_expectancy_yrs": "Life Exp (yrs)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_its_tab(panel: pd.DataFrame) -> None:
    st.subheader("Interrupted Time-Series (ITS) Quasi-Experimental Policy Evaluation")
    st.caption("Segmented OLS regression with Newey-West HAC standard errors (maxlags=1)")

    c1, c2, c3 = st.columns(3)
    iso_options = sorted(EU27_NAMES.keys(), key=lambda k: EU27_NAMES[k])
    selected_iso = c1.selectbox(
        "Member State",
        iso_options,
        index=iso_options.index("HU"),
        format_func=lambda k: f"{EU27_NAMES[k]} ({k})",
    )
    outcome_map = {
        "preventable_mortality_sdr": "Preventable Mortality SDR (/100k)",
        "treatable_mortality_sdr": "Treatable Mortality SDR (/100k)",
        "life_expectancy_yrs": "Life Expectancy at Birth (Years)",
    }
    selected_outcome = c2.selectbox("Health Outcome", list(outcome_map.keys()), format_func=lambda k: outcome_map[k])
    break_year = c3.slider("Structural Break / Policy Year", min_value=2015, max_value=2021, value=2020)

    try:
        res = fit_its_segmented_regression(panel, iso2=selected_iso, outcome=selected_outcome, intervention_year=break_year)
    except ValueError as err:
        st.warning(str(err))
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pre-Intervention Slope", f"{res.pre_trend_slope:+.2f} / yr")
    m2.metric("Immediate Level Shift", f"{res.level_change:+.2f}", f"p = {res.level_change_pval:.3f}")
    m3.metric("Post-Break Slope Change", f"{res.slope_change:+.2f} / yr", f"p = {res.slope_change_pval:.3f}")
    m4.metric("2023 Observed vs Counterfactual", f"{res.observed_2023:.1f}", f"CF: {res.counterfactual_2023:.1f}")

    df = res.df_fitted
    post = df[df["year"] >= break_year]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["year"],
            y=df[selected_outcome],
            mode="markers",
            name="Observed Eurostat Data",
            marker=dict(color="#1e293b", size=9),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["year"],
            y=df["fitted"],
            mode="lines",
            name="Segmented ITS Fit",
            line=dict(color="#0284c7", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=post["year"],
            y=post["counterfactual"],
            mode="lines",
            name="Pre-Break Counterfactual",
            line=dict(color="#dc2626", dash="dash", width=2.5),
        )
    )
    fig.add_vline(x=break_year - 0.5, line_dash="dot", line_color="#64748b")
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=outcome_map[selected_outcome],
        height=460,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_hta_tab() -> None:
    st.subheader("Health Technology Assessment (HTA): Cohort Markov Cost-Effectiveness Model")
    st.caption("Evaluation of Point-of-Care Rapid Diagnostic Triage vs. Standard Empiric Hospital Care")

    with st.sidebar:
        st.header("HTA Model Parameters")
        device_cost = st.slider("Diagnostic Test Cost (EUR / patient)", 20, 300, 85, step=5)
        rr_comp = st.slider("Relative Risk of Acute Complication", 0.40, 0.95, 0.68, step=0.02)
        p_comp_std = st.slider("Baseline Complication Rate (Std Care)", 0.05, 0.40, 0.18, step=0.01)
        comp_cost = st.slider("Acute Complication Hospital Cost (EUR)", 1500, 10000, 4800, step=200)
        wtp = st.select_slider("Willingness-to-Pay Threshold (EUR / QALY)", options=[20000, 25000, 30000, 40000, 50000], value=30000)
        horizon = st.slider("Time Horizon (Years)", 5, 20, 10)
        disc_rate = st.selectbox("Annual Discount Rate", [0.015, 0.03, 0.035, 0.05], index=2)

    params = HTAParams(
        horizon_years=horizon,
        discount_rate=disc_rate,
        wtp_threshold_eur=float(wtp),
        device_cost_per_patient=float(device_cost),
        complication_hospital_cost=float(comp_cost),
        p_complication_std=float(p_comp_std),
        rr_complication_device=float(rr_comp),
    )

    base = evaluate_base_case(params)
    psa = run_probabilistic_sensitivity(params, n_sim=1000, seed=42)
    prob_ce = (psa["nmb"] > 0).mean() * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Incremental Cost", f"EUR {base['inc_cost']:+,.2f}")
    k2.metric("Incremental QALYs", f"{base['inc_qaly']:+.4f}")
    icer_label = "DOMINANT" if (base["inc_cost"] < 0 and base["inc_qaly"] > 0) else f"EUR {base['icer']:,.0f}/QALY"
    k3.metric("ICER Status", icer_label)
    k4.metric("Prob. Cost-Effective (PSA)", f"{prob_ce:.1f}%", f"NMB: EUR {base['nmb']:+,.0f}")

    col_left, col_right = st.columns(2)

    with col_left:
        fig_psa = px.scatter(
            psa,
            x="inc_qaly",
            y="inc_cost",
            opacity=0.5,
            labels={"inc_qaly": "Incremental QALYs", "inc_cost": "Incremental Cost (EUR)"},
            title="Probabilistic Sensitivity Analysis (1,000 Monte Carlo Draws)",
            height=420,
        )
        q_line = np.linspace(min(0.0, psa["inc_qaly"].min()), psa["inc_qaly"].max(), 50)
        fig_psa.add_trace(
            go.Scatter(
                x=q_line,
                y=q_line * wtp,
                mode="lines",
                name=f"WTP (EUR {wtp:,}/QALY)",
                line=dict(color="#059669", dash="dash"),
            )
        )
        fig_psa.add_hline(y=0, line_color="#64748b", line_width=1)
        fig_psa.add_vline(x=0, line_color="#64748b", line_width=1)
        st.plotly_chart(fig_psa, use_container_width=True)

    with col_right:
        trace_dev = base["trace_dev"].melt(id_vars=["Cycle"], var_name="Health State", value_name="Cohort Proportion")
        fig_trace = px.area(
            trace_dev,
            x="Cycle",
            y="Cohort Proportion",
            color="Health State",
            title="Markov Cohort State Occupancy (Rapid Triage Arm)",
            color_discrete_map={
                "Healthy": "#0f766e",
                "Chronic_Sequelae": "#f59e0b",
                "Dead": "#64748b",
            },
            height=420,
        )
        st.plotly_chart(fig_trace, use_container_width=True)


def main() -> None:
    st.title("European Health System Efficiency, Policy Impact & HTA Analytics")
    panel = load_panel()

    tab1, tab2, tab3 = st.tabs([
        "1. EU Health Disparities & Efficiency",
        "2. Interrupted Time-Series Policy Lab",
        "3. HTA Markov Cost-Effectiveness",
    ])

    with tab1:
        render_disparities_tab(panel)
    with tab2:
        render_its_tab(panel)
    with tab3:
        render_hta_tab()


if __name__ == "__main__":
    main()
