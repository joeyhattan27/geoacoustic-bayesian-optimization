import csv
import hashlib
import json
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from . import config
from .acoustics import AcousticModel
from .objective import GeoacousticObjective
from .optimization import (
    evaluate_objective,
    run_ensemble_bo,
    run_single_kernel_bo,
    run_sobol_search,
    sobol_points,
)


METHODS = (
    "sobol",
    "rbf_logei",
    "matern15_logei",
    "matern25_logei",
    "ensemble_uniform_logei",
)

METHOD_LABELS = {
    "sobol": "Sobol",
    "rbf_logei": "RBF + LogEI",
    "matern15_logei": "Matérn 3/2 + LogEI",
    "matern25_logei": "Matérn 5/2 + LogEI",
    "ensemble_uniform_logei": "Uniform ensemble + LogEI",
}

PARAMETER_LABELS = {
    "source_range_km": "Source range",
    "source_depth_m": "Source depth",
    "array_tilt_deg": "Array tilt",
    "water_depth_m": "Water depth",
    "sediment_thickness_m": "Sediment thickness",
    "sediment_top_speed_mps": "Sediment top speed",
    "sediment_speed_change_mps": "Sediment speed change",
}


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def write_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_metadata(directory):
    source_dir = Path(__file__).parent
    files = ("config.py", "optimization.py", "main.py")

    metadata = {
        "experiment": "final_heldout_evaluation",
        "seeds": config.SEEDS,
        "total_evaluations": config.TOTAL_EVALS,
        "warmup_evaluations": config.N_INIT,
        "bo_iterations": config.TOTAL_EVALS - config.N_INIT,
        "acquisition": config.ACQUISITION,
        "fit_method": config.FIT_METHOD,
        "ensemble_weights": config.ENSEMBLE_WEIGHTS.tolist(),
        "num_restarts": config.NUM_RESTARTS,
        "raw_samples": config.RAW_SAMPLES,
        "parameter_names": list(config.PARAMETER_NAMES),
        "lower_bounds": config.LOWER_BOUNDS.tolist(),
        "upper_bounds": config.UPPER_BOUNDS.tolist(),
        "truth": config.TRUE_PARAMETERS.tolist(),
        "versions": {
            name: package_version(name)
            for name in (
                "torch",
                "botorch",
                "gpytorch",
                "numpy",
                "tritonoa",
            )
        },
        "code_sha256": {
            name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
            for name in files
        },
    }

    (directory / "settings.json").write_text(json.dumps(metadata, indent=2))


def save_run(directory, seed, method, result):
    seed_dir = directory / f"seed_{seed}"
    seed_dir.mkdir(exist_ok=True)

    arrays = {
        key: value.detach().cpu().numpy()
        for key, value in result.items()
    }
    np.savez(seed_dir / f"{method}.npz", **arrays)

    parameters = config.normalized_to_physical(arrays["x"])
    rows = []

    for i, parameter in enumerate(parameters):
        row = {
            "evaluation": i + 1,
            "objective": float(arrays["y"][i, 0]),
            "best_objective": float(arrays["best_curve"][i]),
        }
        row.update(dict(zip(config.PARAMETER_NAMES, parameter)))
        rows.append(row)

    write_csv(seed_dir / f"{method}_observations.csv", rows)


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def final_values(runs):
    return np.array([
        run["best_curve"][-1].item()
        for run in runs
    ])


def geometric_mean(values):
    values = np.clip(values, 1e-16, None)
    return float(np.exp(np.mean(np.log(values))))


def method_metrics(values):
    metrics = {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "geometric_mean": geometric_mean(values),
        "mean_log10": float(np.mean(np.log10(np.clip(values, 1e-16, None)))),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "std": float(values.std()),
    }

    for threshold in config.SUCCESS_THRESHOLDS:
        label = f"fraction_below_{threshold:.0e}".replace("-", "m")
        metrics[label] = float(np.mean(values <= threshold))

    return metrics


def best_parameters(result):
    index = result["y"].argmin().item()
    normalized = result["x"][index].detach().cpu().numpy()
    return config.normalized_to_physical(normalized)


def parameter_errors(parameters):
    absolute = np.abs(parameters - config.TRUE_PARAMETERS)
    span = config.UPPER_BOUNDS - config.LOWER_BOUNDS
    normalized = absolute / span
    return absolute, normalized


def plot_convergence(results, path):
    fig, ax = plt.subplots()

    for method, runs in results.items():
        curves = np.stack([
            run["best_curve"].detach().cpu().numpy()
            for run in runs
        ])
        lower, median, upper = np.quantile(curves, [0.25, 0.5, 0.75], axis=0)
        evaluations = np.arange(1, curves.shape[1] + 1)

        ax.plot(evaluations, median, label=METHOD_LABELS[method])
        ax.fill_between(evaluations, lower, upper, alpha=0.10)

    ax.axvline(
        config.N_INIT,
        linestyle="--",
        linewidth=1,
        label="BO starts",
    )
    ax.set_yscale("log")
    ax.set(
        xlabel="Objective evaluations",
        ylabel="Best Bartlett mismatch",
        title="Held-out convergence (50 seeds)",
    )
    ax.legend()
    save_figure(fig, path)


def plot_final_objectives(results, path):
    fig, ax = plt.subplots()
    values = [final_values(results[method]) for method in METHODS]

    ax.boxplot(values, tick_labels=[METHOD_LABELS[m] for m in METHODS], showfliers=True)
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=25)
    ax.set(
        ylabel="Final best Bartlett mismatch",
        title="Held-out objective distribution (50 seeds)",
    )
    save_figure(fig, path)


def plot_parameter_error(parameter_rows, path):
    fig, ax = plt.subplots()
    x = np.arange(len(config.PARAMETER_NAMES))

    for method in METHODS:
        rows = [row for row in parameter_rows if row["method"] == method]
        values = [row["normalized_mae"] for row in rows]
        ax.plot(x, values, marker="o", label=METHOD_LABELS[method])

    ax.set_xticks(x)
    ax.set_xticklabels([PARAMETER_LABELS[name] for name in config.PARAMETER_NAMES], rotation=25, ha="right")
    ax.set(
        ylabel="Mean absolute error / parameter range",
        title="Normalized parameter recovery error",
    )
    ax.legend()
    save_figure(fig, path)


def save_summaries(directory, results, summary_rows):
    method_rows = []

    for method in METHODS:
        values = final_values(results[method])
        runtimes = np.array([
            row["seconds"]
            for row in summary_rows
            if row["method"] == method
        ])

        normalized_errors = np.array([
            row["mean_normalized_parameter_error"]
            for row in summary_rows
            if row["method"] == method
        ])
        normalized_rmse = np.array([
            row["rmse_normalized_parameter_error"]
            for row in summary_rows
            if row["method"] == method
        ])

        row = {"method": method}
        row.update(method_metrics(values))
        row["mean_seconds"] = float(runtimes.mean())
        row["median_seconds"] = float(np.median(runtimes))
        row["mean_normalized_parameter_error"] = float(normalized_errors.mean())
        row["median_normalized_parameter_error"] = float(np.median(normalized_errors))
        row["mean_normalized_parameter_rmse"] = float(normalized_rmse.mean())
        method_rows.append(row)

    write_csv(directory / "method_summary.csv", method_rows)

    pairwise_rows = []

    for i, method_a in enumerate(METHODS):
        for method_b in METHODS[i + 1:]:
            a = final_values(results[method_a])
            b = final_values(results[method_b])
            pairwise_rows.append({
                "method_a": method_a,
                "method_b": method_b,
                "a_wins": int((a < b).sum()),
                "b_wins": int((b < a).sum()),
                "ties": int((a == b).sum()),
                "median_log10_ratio_a_over_b": float(
                    np.median(
                        np.log10(np.clip(a, 1e-16, None))
                        - np.log10(np.clip(b, 1e-16, None))
                    )
                ),
            })

    write_csv(directory / "pairwise_summary.csv", pairwise_rows)

    parameter_rows = []

    for method in METHODS:
        estimates = np.stack([
            best_parameters(run)
            for run in results[method]
        ])
        absolute = np.abs(estimates - config.TRUE_PARAMETERS)
        normalized = absolute / (config.UPPER_BOUNDS - config.LOWER_BOUNDS)

        for i, name in enumerate(config.PARAMETER_NAMES):
            parameter_rows.append({
                "method": method,
                "parameter": name,
                "truth": float(config.TRUE_PARAMETERS[i]),
                "mae": float(absolute[:, i].mean()),
                "median_absolute_error": float(np.median(absolute[:, i])),
                "normalized_mae": float(normalized[:, i].mean()),
                "normalized_median_absolute_error": float(
                    np.median(normalized[:, i])
                ),
            })

    write_csv(directory / "parameter_summary.csv", parameter_rows)
    plot_parameter_error(parameter_rows, directory / "parameter_error.png")


def main():
    directory = (
        config.RESULTS_DIR
        / datetime.now().strftime("final_evaluation_%Y%m%d_%H%M%S_%f")
    )
    directory.mkdir(parents=True)
    save_metadata(directory)

    print(f"Saving this run to: {directory}", flush=True)
    print("Generating the synthetic observation...", flush=True)

    objective = GeoacousticObjective(AcousticModel())
    truth_value = objective.evaluate_physical(config.TRUE_PARAMETERS)

    print(f"Objective at simulated truth: {truth_value:.3e}", flush=True)

    if not np.isfinite(truth_value) or abs(truth_value) > 1e-8:
        raise RuntimeError("Truth self-match failed; stop before BO")

    results = {method: [] for method in METHODS}
    summary = []

    for seed in config.SEEDS:
        print(f"\nSeed {seed}", flush=True)

        full_x = sobol_points(config.TOTAL_EVALS, seed)

        warmup_start = perf_counter()
        warmup_y = evaluate_objective(objective, full_x[:config.N_INIT])
        warmup_seconds = perf_counter() - warmup_start

        tail_start = perf_counter()
        tail_y = evaluate_objective(objective, full_x[config.N_INIT:])
        tail_seconds = perf_counter() - tail_start

        sobol_y = np.concatenate([
            warmup_y.detach().cpu().numpy(),
            tail_y.detach().cpu().numpy(),
        ])
        sobol_y = warmup_y.new_tensor(sobol_y)

        method_specs = (
            ("sobol", None),
            ("rbf_logei", "rbf"),
            ("matern15_logei", "matern15"),
            ("matern25_logei", "matern25"),
            ("ensemble_uniform_logei", "ensemble"),
        )

        for method, kernel in method_specs:
            print(f"  Running {method}...", flush=True)

            if method == "sobol":
                result = run_sobol_search(full_x, sobol_y)
                seconds = warmup_seconds + tail_seconds
            else:
                start = perf_counter()

                if kernel == "ensemble":
                    result = run_ensemble_bo(
                        objective,
                        seed,
                        initial_x=full_x[:config.N_INIT],
                        initial_y=warmup_y,
                    )
                else:
                    result = run_single_kernel_bo(
                        objective,
                        kernel_name=kernel,
                        seed=seed,
                        initial_x=full_x[:config.N_INIT],
                        initial_y=warmup_y,
                    )

                seconds = warmup_seconds + (perf_counter() - start)

            save_run(directory, seed, method, result)
            results[method].append(result)

            parameters = best_parameters(result)
            absolute_error, normalized_error = parameter_errors(parameters)
            value = result["best_curve"][-1].item()

            row = {
                "seed": seed,
                "method": method,
                "best_objective": value,
                "seconds": seconds,
                "mean_normalized_parameter_error": float(normalized_error.mean()),
                "rmse_normalized_parameter_error": float(
                    np.sqrt(np.mean(normalized_error ** 2))
                ),
            }
            row.update(dict(zip(config.PARAMETER_NAMES, parameters)))

            for name, error in zip(config.PARAMETER_NAMES, absolute_error):
                row[f"abs_error_{name}"] = float(error)

            summary.append(row)
            write_csv(directory / "summary.csv", summary)

            print(
                f"  {method:25s} final best = {value:.6e}",
                flush=True,
            )

    plot_convergence(results, directory / "convergence.png")
    plot_final_objectives(results, directory / "final_objectives.png")
    save_summaries(directory, results, summary)

    print("\nFinal held-out evaluation: lower is better")

    for method in METHODS:
        metrics = method_metrics(final_values(results[method]))
        print(
            f"{method:25s} "
            f"median={metrics['median']:.6e} "
            f"geomean={metrics['geometric_mean']:.6e} "
            f"q90={metrics['q90']:.6e}"
        )

    print(f"\nResults saved to: {directory}")


if __name__ == "__main__":
    main()
