import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from torch.quasirandom import SobolEngine

from .config import (
    DIM,
    ENSEMBLE_WEIGHTS,
    KERNELS,
    N_INIT,
    NUM_RESTARTS,
    RAW_SAMPLES,
    TOTAL_EVALS,
)


DTYPE = torch.double
UNIT_BOUNDS = torch.stack([
    torch.zeros(DIM, dtype=DTYPE),
    torch.ones(DIM, dtype=DTYPE),
])


def sobol_points(num_points, seed):
    engine = SobolEngine(dimension=DIM, scramble=True, seed=seed)
    return engine.draw(num_points).to(dtype=DTYPE)


def evaluate_objective(objective, x):
    values = objective(x.detach().cpu().numpy())
    y = torch.as_tensor(values, dtype=DTYPE)

    if y.shape != (len(x), 1) or not torch.isfinite(y).all():
        raise ValueError("The objective must return finite values with shape n x 1")

    return y


def best_curve(y):
    return torch.cummin(y.squeeze(-1), dim=0).values


def run_sobol_search(x, y):
    if x.shape != (TOTAL_EVALS, DIM) or y.shape != (TOTAL_EVALS, 1):
        raise ValueError("Sobol baseline must contain TOTAL_EVALS points")

    return {
        "x": x.clone(),
        "y": y.clone(),
        "best_curve": best_curve(y),
    }


def build_kernel(kernel_name):
    prior = GammaPrior(3.0, 6.0)

    if kernel_name == "rbf":
        base = RBFKernel(ard_num_dims=DIM, lengthscale_prior=prior)
    elif kernel_name in ("matern15", "matern25"):
        nu = 1.5 if kernel_name == "matern15" else 2.5
        base = MaternKernel(
            nu=nu,
            ard_num_dims=DIM,
            lengthscale_prior=prior,
        )
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")

    return ScaleKernel(base)


def standardize_scores(y_objective):
    score = -y_objective
    mean = score.mean()
    scale = score.std().clamp_min(1e-8)
    return (score - mean) / scale, mean, scale


def fit_gp(x, y_standardized, kernel_name):
    likelihood = GaussianLikelihood(
        noise_constraint=Interval(1e-8, 1e-3)
    )

    model = SingleTaskGP(
        train_X=x,
        train_Y=y_standardized,
        covar_module=build_kernel(kernel_name),
        likelihood=likelihood,
    )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    try:
        fit_gpytorch_mll(mll)
    except Exception as error:
        raise RuntimeError(f"GP fitting failed for {kernel_name}") from error

    model.eval()
    model.likelihood.eval()
    return model


def fit_ensemble(x, y):
    y_standardized, score_mean, score_scale = standardize_scores(y)
    models = [
        fit_gp(x, y_standardized, kernel_name)
        for kernel_name in KERNELS
    ]
    return models, score_mean, score_scale


def best_standardized_score(y, score_mean, score_scale):
    return ((-y - score_mean) / score_scale).max()


def optimize_acquisition(acquisition):
    candidate, _ = optimize_acqf(
        acq_function=acquisition,
        bounds=UNIT_BOUNDS,
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )
    return candidate.detach()


def initial_data(initial_x, initial_y):
    if initial_x.shape != (N_INIT, DIM) or initial_y.shape != (N_INIT, 1):
        raise ValueError("Shared warmup must have shape N_INIT x DIM and N_INIT x 1")
    return initial_x.clone(), initial_y.clone()


def run_single_kernel_bo(
    objective,
    kernel_name,
    seed,
    initial_x,
    initial_y,
):
    if kernel_name not in KERNELS:
        raise ValueError(f"Unknown kernel: {kernel_name}")

    torch.manual_seed(seed)
    x, y = initial_data(initial_x, initial_y)

    while len(x) < TOTAL_EVALS:
        y_standardized, score_mean, score_scale = standardize_scores(y)
        model = fit_gp(x, y_standardized, kernel_name)
        best_f = best_standardized_score(y, score_mean, score_scale)
        acquisition = LogExpectedImprovement(model=model, best_f=best_f)
        candidate = optimize_acquisition(acquisition)
        y_new = evaluate_objective(objective, candidate)

        x = torch.cat([x, candidate], dim=0)
        y = torch.cat([y, y_new], dim=0)

    return {
        "x": x,
        "y": y,
        "best_curve": best_curve(y),
    }


class EnsembleLogEI(AcquisitionFunction):
    def __init__(self, models, weights, best_f):
        super().__init__(model=models[0])
        self.weights = weights
        self.acquisitions = [
            LogExpectedImprovement(model=model, best_f=best_f)
            for model in models
        ]

    def forward(self, x):
        log_ei = torch.stack([
            acquisition(x)
            for acquisition in self.acquisitions
        ], dim=0)

        log_weights = self.weights.clamp_min(1e-12).log()
        log_weights = log_weights.view(-1, *([1] * (log_ei.ndim - 1)))
        return torch.logsumexp(log_weights + log_ei, dim=0)


def run_ensemble_bo(objective, seed, initial_x, initial_y):
    torch.manual_seed(seed)
    x, y = initial_data(initial_x, initial_y)
    weights = torch.as_tensor(ENSEMBLE_WEIGHTS, dtype=DTYPE)

    while len(x) < TOTAL_EVALS:
        models, score_mean, score_scale = fit_ensemble(x, y)
        best_f = best_standardized_score(y, score_mean, score_scale)
        acquisition = EnsembleLogEI(models, weights, best_f)
        candidate = optimize_acquisition(acquisition)
        y_new = evaluate_objective(objective, candidate)

        x = torch.cat([x, candidate], dim=0)
        y = torch.cat([y, y_new], dim=0)

    return {
        "x": x,
        "y": y,
        "best_curve": best_curve(y),
    }
