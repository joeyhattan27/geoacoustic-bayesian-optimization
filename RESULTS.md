# Final Results

## Experimental protocol

The final algorithm was frozen before inspecting Seeds 90-139. These 50 seeds were reserved as the held-out evaluation set after all weighting, acquisition, fitting, architecture, and warmup decisions had been made.

The final experiment used:

- 100 objective evaluations per method
- 64 shared Sobol warmup points
- 36 BO iterations
- Log Expected Improvement
- `fit_gpytorch_mll`
- 40 acquisition restarts
- 1024 raw acquisition samples
- identical initial data across BO methods within each seed

The Python 3.13 cleanup run is treated as the canonical final result.

## Objective optimization

| Method | Mean | Median | Geometric mean | q75 | q90 | <= 1e-3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sobol | 8.163e-2 | 8.156e-2 | 6.726e-2 | 1.128e-1 | 1.480e-1 | 0% |
| RBF + LogEI | 3.303e-3 | 8.850e-4 | 9.321e-4 | 3.221e-3 | 8.562e-3 | 58% |
| **Matérn 3/2 + LogEI** | **1.889e-3** | **4.105e-4** | **3.990e-4** | 2.124e-3 | 7.505e-3 | **70%** |
| Matérn 5/2 + LogEI | 2.646e-3 | 5.771e-4 | 5.558e-4 | 2.042e-3 | **5.080e-3** | 58% |
| Uniform ensemble + LogEI | 2.439e-3 | 6.212e-4 | 4.584e-4 | **1.368e-3** | 5.944e-3 | 68% |

All BO methods beat Sobol on every held-out seed.

### Pairwise wins

- Matérn 3/2 beat RBF on 39/50 seeds.
- Matérn 3/2 beat Matérn 5/2 on 31/50 seeds.
- Matérn 3/2 beat the uniform ensemble on 29/50 seeds.
- The uniform ensemble beat Matérn 5/2 on 29/50 seeds.
- The uniform ensemble beat RBF on 32/50 seeds.

The pairwise comparisons reinforce that there is no universally dominant kernel, even though Matérn 3/2 produced the strongest aggregate objective performance.

## Parameter recovery

Mean normalized error averages the absolute parameter error after dividing each dimension by its search-range width.

| Method | Mean normalized error | Mean normalized RMSE |
| --- | ---: | ---: |
| Sobol | 0.1827 | 0.2434 |
| RBF + LogEI | 0.1295 | 0.1815 |
| Matérn 3/2 + LogEI | 0.1213 | 0.1756 |
| Matérn 5/2 + LogEI | 0.1247 | 0.1780 |
| **Uniform ensemble + LogEI** | **0.1188** | **0.1690** |

The source range, source depth, and array tilt were consistently recovered much more accurately than water depth and the sediment parameters. This indicates that low Bartlett mismatch does not imply equal identifiability across all seven physical dimensions.

The ensemble's strongest relative parameter result was sediment sound-speed change, where its normalized MAE was about 0.216 compared with 0.237 for Matérn 3/2 and 0.249 for Matérn 5/2.

## Runtime

Approximate mean per-seed runtimes from the canonical run:

| Method | Mean runtime |
| --- | ---: |
| Sobol | 14.0 s |
| RBF + LogEI | 27.9 s |
| Matérn 3/2 + LogEI | 23.3 s |
| Matérn 5/2 + LogEI | 25.5 s |
| Uniform ensemble + LogEI | 51.6 s |

The ensemble therefore cost roughly twice as much surrogate-model time because three GPs were fitted during each BO iteration.

## Development path

The final method was selected through controlled ablations rather than one large hyperparameter search:

1. **Initial kernel comparison:** fixed RBF, Matérn 3/2, Matérn 5/2, and a predictive-likelihood ensemble.
2. **Diagnostics:** logged kernel weights, entropy, predictive calibration, and hyperparameters; found that early likelihood observations could create path-dependent weight collapse.
3. **Weighting ablation:** tested uniform, tempered predictive likelihood, and shrinkage. Tempering alone did not solve the problem; uniform weighting was unexpectedly competitive.
4. **Evidence ablation:** tested full-dataset marginal likelihood and rolling predictive evidence. Evidence weights stayed nearly uniform and did not improve robustness consistently.
5. **Acquisition ablation:** UCB vs. LogEI; LogEI was retained after fresh-seed validation.
6. **GP fitting validation:** AdamW vs. `fit_gpytorch_mll`; the latter was retained for the final ensemble and simplified final implementation.
7. **Architecture validation:** uniform ensemble vs. evidence-weighted ensemble; uniform averaging was more robust.
8. **Warmup sensitivity:** 32, 48, and 64 initial points; 64 was retained for the final ensemble and preserved the paper's 100/64 evaluation structure.
9. **Held-out evaluation:** the final frozen methods were evaluated on untouched Seeds 90-139.

## Conclusion

Bayesian optimization was highly effective for the simulated SWellEx-96 inversion, but the final study did not support a claim that kernel ensembling universally improves optimization. Matérn 3/2 was the strongest overall objective optimizer on the held-out set. The equal-weight ensemble was competitive and gave the best aggregate physical-parameter recovery, but it did not justify its additional GP-fitting cost solely on objective performance.

The central research finding is therefore a negative but useful one: **kernel diversity can provide a robust hedge, while aggressive or evidence-based online kernel weighting can add instability without a corresponding held-out benefit.**
