# Authorization Saturation in AI-Governed Institutions

Code and figures for my SBP-BRiMS 2026 paper. Built this during my junior year — it's a queueing model for what happens when AI generates decisions faster than humans can review them.

— Shauray Gupta, SM Shetty International School, Mumbai

## Quick start

```bash
pip install -r requirements.txt
python3 cli.py --mode all
```

That runs the sweep, intervention tests, and sensitivity analysis. Takes about 10 seconds.

## What's in here

### The simulation (`simulate_authorization_saturation.py`)

This is the main thing. It's a discrete-event simulation of an M/M/c queue where the servers switch from "substantive" to "ceremonial" mode when the queue gets too long. The switching threshold is T = k*c where k is what I call the institutional tolerance parameter.

The key output is the quality function Q̄(ρ_s) — how review quality drops as utilization increases.

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

Drops everything in `figures/`. The main ones are `fig2_quality.pdf` and `fig3_interventions.pdf` — those are the ones in the paper. The `supp1` through `supp7` files are supplementary.

### Tests

```bash
python3 test_simulation.py
```

Seven tests — checks that quality is high at low load, drops at high load, capacity helps, sigmoid is smoother than binary, etc.

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

MIT — use it however you want.
