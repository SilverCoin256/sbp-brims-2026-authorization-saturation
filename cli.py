"""CLI entry point for authorization saturation simulation."""
import argparse
import sys
from simulate_authorization_saturation import (
    run_sweep, run_intervention_test, run_sigmoid_sensitivity
)


def main():
    parser = argparse.ArgumentParser(
        description="Authorization Saturation Simulation — SBP-BRiMS 2026"
    )
    parser.add_argument(
        "--mode", choices=["sweep", "intervention", "sensitivity", "all"],
        default="all", help="Simulation mode (default: all)"
    )
    args = parser.parse_args()

    if args.mode in ("sweep", "all"):
        print("=" * 55)
        print("Table 2 Reproduction: Quality sweep across rho_s")
        print("=" * 55)
        run_sweep()

    if args.mode in ("intervention", "all"):
        print("\n" + "=" * 55)
        print("Intervention Validation")
        print("=" * 55)
        run_intervention_test()

    if args.mode in ("sensitivity", "all"):
        print("\n" + "=" * 55)
        print("Sigmoid Sensitivity Analysis")
        print("=" * 55)
        run_sigmoid_sensitivity()

    print("\nDone.")


if __name__ == "__main__":
    main()
