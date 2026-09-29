# Bayesian Optimization for Geoacoustic Inversion

A compact research implementation of Gaussian-process Bayesian optimization for the simulated SWellEx-96 geoacoustic inverse problem. The project reproduces a seven-parameter underwater-acoustics inversion with KRAKENC, then compares RBF, Matérn, and ensemble GP surrogates under a fixed 100-evaluation budget.

The main result is deliberately not a forced ensemble win: on 50 held-out seeds, **Matérn 3/2 + LogEI produced the strongest overall objective performance**, while the **equal-weight ensemble produced the lowest aggregate normalized parameter-recovery error**. More complicated adaptive weighting rules were tested during development but did not justify their added complexity.

![Held-out convergence](docs/figures/convergence.png)

## Problem setup

The implementation follows the simulated inversion setup from Jenkins, Gerstoft, and Park (2024), *Geoacoustic inversion using Bayesian optimization with a Gaussian process surrogate model*, JASA 156, 812-822. DOI: [10.1121/10.0028177](https://doi.org/10.1121/10.0028177).

The seven optimized parameters are:

| Parameter | Search range | Simulated truth |
| --- | ---: | ---: |
| Source range | 0.8-1.3 km | 1.07 km |
| Source depth | 60-80 m | 71.1 m |
| Array tilt | -3 to 3 deg | 2.0 deg |
| Water depth | 212-222 m | 217 m |
| Sediment thickness | 15-25 m | 23 m |
| Sediment top sound speed | 1560-1580 m/s | 1572.3 m/s |
| Sediment sound-speed change | 10-30 m/s | 20.7 m/s |

Synthetic pressure fields are evaluated at **148, 235, and 388 Hz** on a 21-element vertical line array. The forward model uses KRAKENC through [TritonOA](https://github.com/NeptuneProjects/TritonOA), with the SWellEx-96 CTD profile `i9605.prn` defining the water-column sound-speed profile.

## Final methods

The held-out experiment compares:

- Sobol search
- RBF GP + Log Expected Improvement
- Matérn 3/2 GP + Log Expected Improvement
- Matérn 5/2 GP + Log Expected Improvement
- Equal-weight RBF / Matérn 3/2 / Matérn 5/2 ensemble + Log Expected Improvement

All BO methods use the same experimental controls:

- 100 total objective evaluations
- 64 shared Sobol warmup evaluations
- 36 Bayesian-optimization decisions
- ARD lengthscales with a Gamma(3, 6) prior
- `fit_gpytorch_mll` for GP hyperparameter fitting
- 40 acquisition-function restarts
- 1024 raw acquisition samples
- identical warmup observations across methods within a seed

The final ensemble uses fixed weights of `1/3` for each kernel. Development experiments included predictive-likelihood weighting, tempered updates, shrinkage, full-dataset marginal-likelihood evidence, rolling evidence, UCB vs. LogEI, AdamW vs. `fit_gpytorch_mll`, and 32/48/64-point warmups. Those experiments are summarized in [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md).

## Final held-out results

The final environment uses Python 3.13 and evaluates 50 untouched seeds (90-139).

| Method | Mean objective | Median | Geometric mean | 90th percentile |
| --- | ---: | ---: | ---: | ---: |
| Sobol | 8.163e-2 | 8.156e-2 | 6.726e-2 | 1.480e-1 |
| RBF + LogEI | 3.303e-3 | 8.850e-4 | 9.321e-4 | 8.562e-3 |
| **Matérn 3/2 + LogEI** | **1.889e-3** | **4.105e-4** | **3.990e-4** | 7.505e-3 |
| Matérn 5/2 + LogEI | 2.646e-3 | 5.771e-4 | 5.558e-4 | **5.080e-3** |
| Uniform ensemble + LogEI | 2.439e-3 | 6.212e-4 | 4.584e-4 | 5.944e-3 |

Every BO method beat Sobol on all 50 held-out seeds. Matérn 3/2 beat Matérn 5/2 on 31/50 seeds and beat the ensemble on 29/50. The ensemble beat Matérn 5/2 on 29/50 seeds and RBF on 32/50.

The ensemble did, however, produce the lowest overall mean normalized parameter-recovery error:

| Method | Mean normalized parameter error |
| --- | ---: |
| Sobol | 0.1827 |
| RBF + LogEI | 0.1295 |
| Matérn 3/2 + LogEI | 0.1213 |
| Matérn 5/2 + LogEI | 0.1247 |
| **Uniform ensemble + LogEI** | **0.1188** |

![Final objective distributions](docs/figures/final_objectives.png)

![Normalized parameter recovery error](docs/figures/parameter_recovery.png)

Detailed aggregate tables are in [`results/final/`](results/final/), and the research interpretation is in [RESULTS.md](RESULTS.md).

## Project structure

```text
bayesian-optimization/
├── data/
│   └── i9605.prn
├── docs/
│   └── figures/
├── results/
│   └── final/
├── src/
│   └── bayesian_optimization/
│       ├── __init__.py
│       ├── acoustics.py
│       ├── config.py
│       ├── main.py
│       ├── objective.py
│       └── optimization.py
├── .python-version
├── pyproject.toml
├── README.md
├── RESULTS.md
├── PROJECT_SUMMARY.md
└── INTERVIEW_GUIDE.md
```

## Requirements

KRAKENC from the [Acoustics Toolbox](https://github.com/oalib-acoustics/Acoustics-Toolbox) must be compiled separately and available on `PATH`.

```bash
which krakenc.exe
```

The project expects the SWellEx-96 CTD profile at:

```text
data/i9605.prn
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv python pin 3.13
uv sync
```

Verify the external model and Python package:

```bash
which krakenc.exe
uv run python -c "import bayesian_optimization; print(bayesian_optimization.__file__)"
```

Run the final experiment:

```bash
uv run bayesian-optimization
```

Each execution writes a timestamped directory under `results/` containing seed-level trajectories, aggregate CSV summaries, figures, dependency versions, and source-code hashes.

## Research conclusion

Bayesian optimization reduced the acoustic mismatch by roughly two orders of magnitude relative to space-filling Sobol search under the same evaluation budget. Kernel choice mattered: Matérn 3/2 gave the best overall held-out objective performance, while Matérn 5/2 had the strongest 90th-percentile robustness. The equal-weight ensemble remained competitive and slightly improved aggregate recovery of the physical parameter vector, but it required fitting three GPs per BO iteration and did not consistently outperform the best fixed kernel.

The main methodological lesson was that **model averaging helped, but more complicated online kernel-selection rules did not** on this noiseless benchmark.

## References

- W. F. Jenkins, P. Gerstoft, and Y. Park, “Geoacoustic inversion using Bayesian optimization with a Gaussian process surrogate model,” *JASA* 156, 812-822 (2024). [doi:10.1121/10.0028177](https://doi.org/10.1121/10.0028177)
- [NeptuneProjects/BOGP](https://github.com/NeptuneProjects/BOGP) — authors' public Bayesian-optimization implementation
- [NeptuneProjects/TritonOA](https://github.com/NeptuneProjects/TritonOA) — Python interface used to run the acoustic model
- [oalib-acoustics/Acoustics-Toolbox](https://github.com/oalib-acoustics/Acoustics-Toolbox) — KRAKENC propagation model
