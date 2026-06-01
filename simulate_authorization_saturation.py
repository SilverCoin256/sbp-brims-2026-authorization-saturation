"""
Authorization Saturation Simulation
Coupled M/M/c queue with state-dependent service rates.
Reproduces Table 2 in SBP-BRiMS 2026 submission.

Usage:
    python simulate_authorization_saturation.py

Requirements: Python 3.8+, numpy only.
"""

import numpy as np
import sys

def simulate_authorization_queue(
    lam: float,
    mu_s: float = 1.0,
    alpha: float = 3.0,
    c: int = 5,
    k: float = 1.0,
    q_s: float = 1.0,
    q_c: float = 0.25,
    T_sim: float = 50_000,
    T_warmup: float = 5_000,
    seed: int = 42
) -> dict:
    """
    Discrete-event simulation of state-dependent M/M/c authorization queue.

    Parameters
    ----------
    lam     : arrival rate (lambda)
    mu_s    : per-server substantive service rate
    alpha   : ceremonial speedup factor (mu_c = alpha * mu_s)
    c       : number of reviewers
    k       : tolerance multiplier (threshold T = k*c)
    q_s     : authorization quality in substantive mode
    q_c     : authorization quality in ceremonial mode
    T_sim   : total simulation time
    T_warmup: warm-up period (excluded from statistics)
    seed    : random seed

    Returns
    -------
    dict with keys: rho_s, mean_N, mean_Q, n_arrivals, n_completions
    """
    rng = np.random.default_rng(seed)
    mu_c = alpha * mu_s
    T_threshold = int(k * c)

    # State variables
    t = 0.0
    N = 0          # number in system (queue + service)
    n_busy = 0     # number of servers busy

    # Statistics accumulators (post warm-up)
    area_N = 0.0
    area_Q = 0.0
    t_last = 0.0
    t_start_stats = T_warmup

    total_arrivals = 0
    total_completions = 0

    # Event heap: (time, type) where type in {'arrival', 'departure'}
    # Use sorted list approach for simplicity
    next_arrival = rng.exponential(1.0 / lam)
    # departure times for each busy server
    departure_times = []

    def current_mu_per_server(n_in_system):
        return mu_c if n_in_system > T_threshold else mu_s

    def schedule_departure(n_in_system):
        rate = current_mu_per_server(n_in_system)
        return t + rng.exponential(1.0 / rate)

    while t < T_sim + T_warmup:
        # Find next event
        next_dep = min(departure_times) if departure_times else float('inf')
        next_event_t = min(next_arrival, next_dep)

        # Accumulate statistics (only post-warmup)
        if t >= t_start_stats:
            dt = next_event_t - max(t, t_start_stats)
            if dt > 0:
                area_N += N * dt
                q_current = q_s if N <= T_threshold else q_c
                area_Q += q_current * dt

        t = next_event_t

        if next_arrival <= next_dep:
            # Arrival event
            total_arrivals += 1
            N += 1
            if n_busy < c:
                n_busy += 1
                departure_times.append(schedule_departure(N))
            next_arrival = t + rng.exponential(1.0 / lam)
        else:
            # Departure event
            total_completions += 1
            N -= 1
            n_busy -= 1
            departure_times.remove(next_dep)
            if N >= c:  # still have queue, start serving next
                n_busy += 1
                departure_times.append(schedule_departure(N))

        if t > T_sim + T_warmup:
            break

    elapsed = T_sim
    mean_N = area_N / elapsed if elapsed > 0 else 0
    mean_Q = area_Q / elapsed if elapsed > 0 else 0

    return {
        "rho_s": lam / (c * mu_s),
        "lam": lam,
        "mean_N": mean_N,
        "mean_Q": mean_Q,
        "n_arrivals": total_arrivals,
        "n_completions": total_completions,
    }


def run_sweep(
    rho_values=None,
    mu_s=1.0, alpha=3.0, c=5, k=1.0,
    q_s=1.0, q_c=0.25,
    T_sim=50_000, T_warmup=5_000
):
    if rho_values is None:
        rho_values = [0.1, 0.2, 0.3, 0.4, 0.45, 0.50, 0.55, 0.6, 0.7, 0.8, 0.9]

    rho_star = k / (1 + k)
    print(f"\nAuthorization Saturation Simulation")
    print(f"Parameters: c={c}, mu_s={mu_s}, alpha={alpha}, k={k}, q_s={q_s}, q_c={q_c}")
    print(f"Theoretical rho* = {rho_star:.3f}")
    print("-" * 65)
    print(f"{'rho_s':>6} {'lambda':>7} {'N_bar':>8} {'Q_bar':>8} {'Regime':<18}")
    print("-" * 65)

    results = []
    for rho_s in rho_values:
        lam = rho_s * c * mu_s
        res = simulate_authorization_queue(
            lam=lam, mu_s=mu_s, alpha=alpha, c=c, k=k,
            q_s=q_s, q_c=q_c, T_sim=T_sim, T_warmup=T_warmup,
            seed=42
        )
        if rho_s < rho_star - 0.08:
            regime = "Substantive"
        elif rho_s > rho_star + 0.08:
            regime = "Ceremonial"
        else:
            regime = "Transition"

        print(f"{res['rho_s']:>6.2f} {res['lam']:>7.2f} {res['mean_N']:>8.2f} "
              f"{res['mean_Q']:>8.3f} {regime:<18}")
        results.append({**res, "regime": regime})

    print("-" * 65)
    return results


def run_intervention_test(rho_ref=0.60, mu_s=1.0, alpha=3.0, c=5, k=1.0,
                          q_s=1.0, q_c=0.25):
    """Test interventions I1-I4 at reference utilization rho_ref."""
    lam_ref = rho_ref * c * mu_s
    print(f"\nIntervention Test at rho_s = {rho_ref:.2f} (lambda = {lam_ref:.2f})")
    print("-" * 55)

    configs = [
        ("Baseline",                    dict(lam=lam_ref, c=c, k=k, q_c=q_c)),
        ("I1: Double capacity (c*2)",   dict(lam=lam_ref, c=c*2, k=k, q_c=q_c)),
        ("I2: Halve throughput (l/2)",  dict(lam=lam_ref/2, c=c, k=k, q_c=q_c)),
        ("I1+I2 (25% each)",            dict(lam=lam_ref*0.75, c=int(c*1.25), k=k, q_c=q_c)),
        ("I3: Raise k to 2.0",          dict(lam=lam_ref, c=c, k=2.0, q_c=q_c)),
        ("I4: Raise q_c to 0.50",       dict(lam=lam_ref, c=c, k=k, q_c=0.50)),
    ]

    print(f"{'Intervention':<30} {'rho_s':>6} {'Q_bar':>8}")
    print("-" * 55)
    for name, cfg in configs:
        r = simulate_authorization_queue(
            lam=cfg['lam'], mu_s=mu_s, alpha=alpha,
            c=cfg['c'], k=cfg['k'], q_s=q_s, q_c=cfg['q_c'],
            seed=42
        )
        print(f"{name:<30} {r['rho_s']:>6.2f} {r['mean_Q']:>8.3f}")
    print("-" * 55)


def simulate_sigmoid_queue(
    lam: float,
    mu_s: float = 1.0,
    alpha: float = 3.0,
    c: int = 5,
    k: float = 1.0,
    q_s: float = 1.0,
    q_c: float = 0.25,
    delta: float = 1.0,
    T_sim: float = 50_000,
    T_warmup: float = 5_000,
    seed: int = 42
) -> dict:
    """Discrete-event simulation with sigmoid (continuous) service rate transition."""
    rng = np.random.default_rng(seed)
    mu_c = alpha * mu_s
    T_threshold = k * c  # float for sigmoid

    t = 0.0
    N = 0
    n_busy = 0
    area_N = 0.0
    area_Q = 0.0
    t_start_stats = T_warmup
    next_arrival = rng.exponential(1.0 / lam)
    departure_times = []

    def sigmoid_mu(n_in_system):
        """Continuous transition: sigmoid centered at T_threshold."""
        s = 1.0 / (1.0 + np.exp(-(n_in_system - T_threshold) / delta))
        return mu_s + (mu_c - mu_s) * s

    def sigmoid_quality(n_in_system):
        """Quality follows same sigmoid transition."""
        s = 1.0 / (1.0 + np.exp(-(n_in_system - T_threshold) / delta))
        return q_s + (q_c - q_s) * s

    def schedule_departure(n_in_system):
        rate = sigmoid_mu(n_in_system)
        return t + rng.exponential(1.0 / rate)

    while t < T_sim + T_warmup:
        next_dep = min(departure_times) if departure_times else float('inf')
        next_event_t = min(next_arrival, next_dep)

        if t >= t_start_stats:
            dt = next_event_t - max(t, t_start_stats)
            if dt > 0:
                area_N += N * dt
                area_Q += sigmoid_quality(N) * dt

        t = next_event_t

        if next_arrival <= next_dep:
            N += 1
            if n_busy < c:
                n_busy += 1
                departure_times.append(schedule_departure(N))
            next_arrival = t + rng.exponential(1.0 / lam)
        else:
            N -= 1
            n_busy -= 1
            departure_times.remove(next_dep)
            if N >= c:
                n_busy += 1
                departure_times.append(schedule_departure(N))

        if t > T_sim + T_warmup:
            break

    elapsed = T_sim
    mean_N = area_N / elapsed if elapsed > 0 else 0
    mean_Q = area_Q / elapsed if elapsed > 0 else 0

    return {"rho_s": lam / (c * mu_s), "lam": lam, "mean_N": mean_N, "mean_Q": mean_Q}


def run_sigmoid_sensitivity():
    """Test sensitivity of threshold result to binary vs sigmoid switching."""
    print("\nSigmoid Sensitivity Analysis (binary vs continuous switching)")
    print("Parameters: c=5, mu_s=1.0, alpha=3.0, k=1.0, q_s=1.0, q_c=0.25")
    print("-" * 70)
    print(f"{'delta':>6} {'rho_s':>7} {'Q_bar':>8} {'Q drop 0.4-0.9':>14} {'rho at Q<0.95':>14}")
    print("-" * 70)

    rho_values = [0.1, 0.2, 0.3, 0.4, 0.45, 0.50, 0.55, 0.6, 0.7, 0.8, 0.9]

    for delta in [0, 1, 2, 5]:  # delta=0 is binary
        results = []
        for rho_s in rho_values:
            lam = rho_s * 5 * 1.0
            if delta == 0:
                r = simulate_authorization_queue(lam=lam, seed=42)
            else:
                r = simulate_sigmoid_queue(lam=lam, delta=delta, seed=42)
            results.append(r)

        # Compute quality drop from rho=0.4 to rho=0.9
        q_at_04 = results[3]['mean_Q']
        q_at_09 = results[10]['mean_Q']
        q_drop = (q_at_04 - q_at_09) * 100

        # Find rho at which Q < 0.95
        rho_below_95 = None
        for r in results:
            if r['mean_Q'] < 0.95:
                rho_below_95 = r['rho_s']
                break

        label = "binary" if delta == 0 else f"{delta}"
        print(f"{label:>6} {'':>7} {'':>8} {q_drop:>13.1f}% {rho_below_95 if rho_below_95 else '>0.9':>14}")

    print("-" * 70)


if __name__ == "__main__":
    run_sweep()
    run_intervention_test()
    run_sigmoid_sensitivity()
    print("\nDone. Results reproduce Tables 2-3 and sensitivity analysis in the paper.")
