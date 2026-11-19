from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class HTAParams:
    horizon_years: int = 10
    discount_rate: float = 0.035          # Standard European HTA discount rate (3.5%)
    wtp_threshold_eur: float = 30000.0    # Willingness-to-pay per QALY gained (EUR)

    # Diagnostic device & clinical pathway costs (EUR)
    device_cost_per_patient: float = 85.0
    uncomplicated_care_cost: float = 620.0
    complication_hospital_cost: float = 4800.0
    chronic_sequelae_annual_cost: float = 2100.0

    # Transition probabilities (annual)
    p_complication_std: float = 0.18
    rr_complication_device: float = 0.68  # 32% relative risk reduction from rapid triage
    p_sequelae_given_comp: float = 0.22
    p_death_acute_comp: float = 0.08
    p_death_sequelae: float = 0.06
    p_death_background: float = 0.012

    # EQ-5D Health-related quality of life utilities
    u_healthy: float = 0.86
    u_complication_disutility: float = 0.18
    u_sequelae: float = 0.64


def _run_cohort_trace(
    p_comp: float,
    upfront_cost: float,
    params: HTAParams,
) -> tuple[float, float, pd.DataFrame]:
    # Markov states: [0: Recovered/Healthy, 1: Post-Acute Sequelae, 2: Dead]
    n_cycles = params.horizon_years
    trace = np.zeros((n_cycles + 1, 3))

    # Cycle 0: Acute episode resolution into post-acute states
    p_death_init = p_comp * params.p_death_acute_comp + (1.0 - p_comp) * params.p_death_background
    p_seq_init = p_comp * params.p_sequelae_given_comp
    p_healthy_init = max(0.0, 1.0 - p_death_init - p_seq_init)

    trace[0] = [p_healthy_init, p_seq_init, p_death_init]

    # Annual transition matrix from Cycle 1 onward
    T = np.array([
        [1.0 - params.p_death_background, 0.0, params.p_death_background],
        [0.0, 1.0 - params.p_death_sequelae, params.p_death_sequelae],
        [0.0, 0.0, 1.0],
    ])

    for t in range(1, n_cycles + 1):
        trace[t] = trace[t - 1] @ T

    # Half-cycle correction for state occupancy
    mid_trace = 0.5 * (trace[:-1] + trace[1:])
    cycles = np.arange(1, n_cycles + 1)
    disc = 1.0 / ((1.0 + params.discount_rate) ** cycles)

    # Episode index cost + discounted chronic care costs
    index_cost = upfront_cost + (1.0 - p_comp) * params.uncomplicated_care_cost + p_comp * params.complication_hospital_cost
    annual_costs = mid_trace[:, 1] * params.chronic_sequelae_annual_cost
    total_cost = float(index_cost + np.sum(annual_costs * disc))

    # Index episode QALY (with acute complication disutility) + discounted follow-up QALYs
    index_qaly = (p_healthy_init * params.u_healthy + p_seq_init * params.u_sequelae) - (p_comp * params.u_complication_disutility * 0.25)
    annual_qalys = mid_trace[:, 0] * params.u_healthy + mid_trace[:, 1] * params.u_sequelae
    total_qaly = float(index_qaly + np.sum(annual_qalys * disc))

    df_trace = pd.DataFrame(trace, columns=["Healthy", "Chronic_Sequelae", "Dead"])
    df_trace["Cycle"] = np.arange(n_cycles + 1)
    return total_cost, total_qaly, df_trace


def evaluate_base_case(params: HTAParams) -> dict:
    cost_std, qaly_std, trace_std = _run_cohort_trace(
        p_comp=params.p_complication_std,
        upfront_cost=0.0,
        params=params,
    )
    p_comp_dev = params.p_complication_std * params.rr_complication_device
    cost_dev, qaly_dev, trace_dev = _run_cohort_trace(
        p_comp=p_comp_dev,
        upfront_cost=params.device_cost_per_patient,
        params=params,
    )

    inc_cost = cost_dev - cost_std
    inc_qaly = qaly_dev - qaly_std
    icer = inc_cost / inc_qaly if inc_qaly != 0 else np.nan
    nmb = (inc_qaly * params.wtp_threshold_eur) - inc_cost

    return {
        "cost_std": cost_std,
        "qaly_std": qaly_std,
        "cost_dev": cost_dev,
        "qaly_dev": qaly_dev,
        "inc_cost": inc_cost,
        "inc_qaly": inc_qaly,
        "icer": icer,
        "nmb": nmb,
        "trace_std": trace_std,
        "trace_dev": trace_dev,
    }


def run_probabilistic_sensitivity(params: HTAParams, n_sim: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Beta draws for probabilities & utilities; Log-Normal for relative risk; Gamma for costs
    p_comp_draws = rng.beta(18, 82, size=n_sim) * (params.p_complication_std / 0.18)
    p_comp_draws = np.clip(p_comp_draws, 0.01, 0.95)

    rr_draws = rng.lognormal(mean=np.log(params.rr_complication_device), sigma=0.12, size=n_sim)
    comp_cost_draws = rng.gamma(shape=25.0, scale=params.complication_hospital_cost / 25.0, size=n_sim)
    seq_cost_draws = rng.gamma(shape=25.0, scale=params.chronic_sequelae_annual_cost / 25.0, size=n_sim)

    records = []
    for i in range(n_sim):
        sim_p = HTAParams(
            horizon_years=params.horizon_years,
            discount_rate=params.discount_rate,
            wtp_threshold_eur=params.wtp_threshold_eur,
            device_cost_per_patient=params.device_cost_per_patient,
            uncomplicated_care_cost=params.uncomplicated_care_cost,
            complication_hospital_cost=float(comp_cost_draws[i]),
            chronic_sequelae_annual_cost=float(seq_cost_draws[i]),
            p_complication_std=float(p_comp_draws[i]),
            rr_complication_device=float(rr_draws[i]),
            p_sequelae_given_comp=params.p_sequelae_given_comp,
            p_death_acute_comp=params.p_death_acute_comp,
            p_death_sequelae=params.p_death_sequelae,
            p_death_background=params.p_death_background,
            u_healthy=params.u_healthy,
            u_complication_disutility=params.u_complication_disutility,
            u_sequelae=params.u_sequelae,
        )
        res = evaluate_base_case(sim_p)
        records.append({
            "sim": i + 1,
            "inc_cost": res["inc_cost"],
            "inc_qaly": res["inc_qaly"],
            "nmb": res["nmb"],
        })

    return pd.DataFrame(records)


def main() -> None:
    params = HTAParams()
    base = evaluate_base_case(params)
    psa = run_probabilistic_sensitivity(params, n_sim=1000)
    prob_ce = (psa["nmb"] > 0).mean() * 100

    print("=== Base-Case HTA Evaluation (10-Year Horizon, 3.5% Discount) ===")
    print(f"Standard Care : EUR {base['cost_std']:,.2f} | {base['qaly_std']:.4f} QALYs")
    print(f"Rapid Triage  : EUR {base['cost_dev']:,.2f} | {base['qaly_dev']:.4f} QALYs")
    print(f"Incremental   : EUR {base['inc_cost']:+,.2f} | {base['inc_qaly']:+.4f} QALYs")
    if base["inc_cost"] < 0 and base["inc_qaly"] > 0:
        print("Status        : DOMINANT (Cost-saving and QALY-gaining)")
    else:
        print(f"ICER          : EUR {base['icer']:,.2f} / QALY")
    print(f"Net Monetary Benefit (@ EUR {params.wtp_threshold_eur:,.0f}/QALY): EUR {base['nmb']:+,.2f}")
    print(f"PSA Probability Cost-Effective (1,000 Monte Carlo sims): {prob_ce:.1f}%")


if __name__ == "__main__":
    main()
