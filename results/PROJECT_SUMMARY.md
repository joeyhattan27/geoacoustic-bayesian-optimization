# Project Summary

## Research question

Can Bayesian optimization recover the seven-dimensional SWellEx-96 geoacoustic environment efficiently, and can combining multiple Gaussian-process kernels improve robustness relative to committing to a single kernel?

## What was implemented

The project couples a real underwater-acoustics forward model with modern Bayesian optimization:

- KRAKENC normal-mode propagation through TritonOA
- simulated SWellEx-96 pressure observations at three frequencies
- a Bartlett matched-field mismatch objective
- RBF, Matérn 3/2, and Matérn 5/2 Gaussian-process surrogates
- Log Expected Improvement acquisition in BoTorch
- an equal-weight mixture of kernel-specific expected improvements
- paired multi-seed experiments with shared Sobol initial designs
- physical parameter-recovery analysis in addition to objective optimization

## How the method evolved

The project began with an adaptive kernel ensemble weighted by one-step predictive likelihood. Multi-seed diagnostics showed that a single early predictive miss could almost eliminate a useful kernel and influence the rest of the short 100-evaluation budget. Tempering and shrinkage reduced this collapse but did not consistently improve optimization.

The next idea was to weight kernels using full-dataset GP marginal likelihood. Those weights stayed very close to uniform, and a direct architecture ablation showed that fixed equal weights were simpler and more robust. Separate acquisition experiments then favored LogEI over UCB, and GP-fitting experiments retained BoTorch's `fit_gpytorch_mll`. A warmup study retained 64 Sobol points before the final algorithm was frozen.

## Final held-out result

The final 50-seed test showed:

- every BO method beat Sobol on every seed;
- Matérn 3/2 + LogEI had the best mean, median, and geometric-mean objective;
- Matérn 5/2 had the best 90th-percentile objective;
- the uniform ensemble remained competitive and had the lowest aggregate normalized parameter-recovery error;
- the ensemble required roughly twice the surrogate-model runtime.

The project therefore ends with a defensible negative result: the ensemble did not consistently beat the best fixed kernel, but the ablation process revealed when and why adaptive weighting failed and where model averaging still helped.

## Technical stack

Python, NumPy, SciPy, PyTorch, GPyTorch, BoTorch, Matplotlib, uv, TritonOA, KRAKENC / Acoustics Toolbox.
