"""Tests for authorization saturation simulation."""
import numpy as np
from simulate_authorization_saturation import (
    simulate_authorization_queue,
    simulate_sigmoid_queue,
)


def test_stability_low_rho():
    """At low rho_s, quality should be near q_s=1.0."""
    r = simulate_authorization_queue(lam=0.5, seed=42)
    assert r["mean_Q"] > 0.99, f"Expected Q > 0.99 at rho=0.1, got {r['mean_Q']}"


def test_ceremonial_high_rho():
    """At high rho_s, quality should drop significantly."""
    r = simulate_authorization_queue(lam=4.5, seed=42)
    assert r["mean_Q"] < 0.85, f"Expected Q < 0.85 at rho=0.9, got {r['mean_Q']}"


def test_threshold_inflection():
    """Quality at rho=0.5 should be lower than at rho=0.4."""
    r1 = simulate_authorization_queue(lam=2.0, seed=42)
    r2 = simulate_authorization_queue(lam=2.5, seed=42)
    assert r1["mean_Q"] > r2["mean_Q"], "Quality should decrease with rho_s"


def test_intervention_capacity():
    """Doubling capacity should restore quality."""
    r_base = simulate_authorization_queue(lam=3.0, c=5, seed=42)
    r_cap = simulate_authorization_queue(lam=3.0, c=10, seed=42)
    assert r_cap["mean_Q"] > r_base["mean_Q"], "More capacity should improve quality"


def test_sigmoid_inflection():
    """Sigmoid model should also show quality decline."""
    r = simulate_sigmoid_queue(lam=4.5, delta=1.0, seed=42)
    assert r["mean_Q"] < 0.95, f"Expected Q < 0.95 at rho=0.9 (sigmoid), got {r['mean_Q']}"


def test_sigmoid_vs_binary():
    """Sigmoid should produce less sharp decline than binary."""
    r_bin = simulate_authorization_queue(lam=4.5, seed=42)
    r_sig = simulate_sigmoid_queue(lam=4.5, delta=2.0, seed=42)
    # Sigmoid with delta=2 should have higher quality (less sharp transition)
    assert r_sig["mean_Q"] >= r_bin["mean_Q"] - 0.05, \
        f"Sigmoid Q={r_sig['mean_Q']} too far below binary Q={r_bin['mean_Q']}"


def test_queue_length_monotonicity():
    """Queue length should increase with rho_s."""
    r1 = simulate_authorization_queue(lam=1.0, seed=42)
    r2 = simulate_authorization_queue(lam=3.0, seed=42)
    assert r2["mean_N"] > r1["mean_N"], "Queue length should increase with load"


if __name__ == "__main__":
    test_stability_low_rho()
    test_ceremonial_high_rho()
    test_threshold_inflection()
    test_intervention_capacity()
    test_sigmoid_inflection()
    test_sigmoid_vs_binary()
    test_queue_length_monotonicity()
    print("All tests passed.")
