# Authorization Saturation in AI-Governed Institutions

Code and figures for my SBP-BRiMS 2026 paper. I built the model during my junior year to test when a queue of AI-generated decisions pushes reviewers from substantive checks to faster, less thorough reviews.

Shauray Gupta, SM Shetty International School, Mumbai

## Quick start

```bash
pip install -r requirements.txt
python3 cli.py --mode all
```

This runs the utilization sweep, intervention tests, and sensitivity analysis.

## What's in here

### The simulation (`simulate_authorization_saturation.py`)

In the discrete-event M/M/c simulation, reviewers switch from substantive to ceremonial review when the queue reaches the model’s threshold. That threshold is `T = k*c`, where `c` is the number of reviewers and `k` is the tolerance parameter.

The simulation estimates mean review quality, Q̄(ρ_s), at different levels of utilization.

### CLI modes

```bash
python3 cli.py --mode sweep         # Table 2: quality vs utilization
python3 cli.py --mode intervention  # Tests I1-I4 at ρ_s = 0.60
python3 cli.py --mode sensitivity   # Binary vs sigmoid switching
```

### Figures

```bash
python3 generate_figures.py
```

The script writes figures to `figures/`. The tracked SVGs for the paper are [review quality](figures/fig2_quality.svg) and [interventions](figures/fig3_interventions.svg). The `supp*` files are supplementary.

### Tests

```bash
python3 test_simulation.py
```

The tests check how review quality responds to load, capacity, and the choice of switching rule.

### Compiling the paper

```bash
pdflatex sbp_brims_2026.tex
bibtex sbp_brims_2026
pdflatex sbp_brims_2026.tex
pdflatex sbp_brims_2026.tex
```

You need `llncs.cls` and `splncs04.bst` (Springer LNCS class files, included).

## Files

```
sbp_brims_2026.tex                  # Paper
references.bib                      # All 24 citations
simulate_authorization_saturation.py # The simulation
generate_figures.py                 # Makes all the plots
cli.py                              # Command-line runner
test_simulation.py                  # Tests
requirements.txt                    # numpy, matplotlib
environment.yml                     # Conda env (Python 3.10)
CITATION.cff                        # Citation info
figures/                            # Output PDFs and SVGs
```

## Default parameters

| Parameter | Value | What it is |
|---|---|---|
| c | 5 | Reviewers |
| μ_s | 1.0 | Substantive rate |
| α | 3.0 | Ceremonial speedup |
| k | 1.0 | Tolerance |
| q_s | 1.0 | Substantive quality |
| q_c | 0.25 | Ceremonial quality |

## The main result

Saturation threshold: **ρ* = k/(1+k)**

At k = 1.0, the institution saturates at ρ* = 0.50 — half its nominal capacity. Adding more reviewers doesn't raise this threshold. Only changing k does.

## License

The project identifies its license as MIT. This checkout omits the license text.
