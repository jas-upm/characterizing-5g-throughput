"""Model evaluation and plotting using the test set."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

import joblib
import shap
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib.legend_handler import HandlerTuple
from ngboost import NGBRegressor
from scipy.special import erf
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


def evaluate_models(
    Xy_train_val_test: Dict[str, Dict[str, Any]],
    config: Dict,
    exp_dir: Path,
) -> None:
    """
    Evaluate trained models on test sets and generate evaluation plots.

    Args:
        Xy_train_val_test: Dictionary containing per-technology test data.
        config: Experiment configuration dictionary.
        exp_dir: Base experiment directory (timestamped).
    """
    eval_dir = exp_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving evaluation outputs to %s", eval_dir)

    data_cfg = config.get("data", {})
    radio_techs = data_cfg.get("radio_access_technologies", [])
    eval_cfg = config.get("eval", {})

    features = data_cfg.get("features", []) + data_cfg.get("temporal_encodings", [])

    metrics_summary: Dict[str, Dict[str, float]] = {}
    shap_dict: Dict[str, Dict[str, str]] = {}

    for tech in radio_techs:
        logger.info("===== Evaluating %s =====", tech)

        test_data = Xy_train_val_test[tech]
        X_test = test_data["X_test"]
        y_test = test_data["y_test"]
        y_norm_mean = test_data["y_norm_mean"]
        y_norm_stdev = test_data["y_norm_stdev"]

        model_dir = exp_dir / "models" / tech
        xgb_model = joblib.load(model_dir / "XGBoost_model.pkl")
        ngb_model = joblib.load(model_dir / "NGBoost_model.pkl")

        # Inverse normalization to natural scale
        y_test_nat = np.expm1(y_test * y_norm_stdev + y_norm_mean)

        # --- XGB predictions ---
        y_pred_xgb = xgb_model.predict(X_test)
        y_pred_xgb_nat = np.expm1(y_pred_xgb * y_norm_stdev + y_norm_mean)

        # --- NGB predictions ---
        y_pred_dist = ngb_model.pred_dist(X_test)
        mu = y_pred_dist.mean().flatten()
        sigma = y_pred_dist.std().flatten()
        y_pred_mean_ng = mu
        y_pred_mean_nat_ng = np.expm1(y_pred_mean_ng * y_norm_stdev + y_norm_mean)

        # --- Compute metrics ---
        metrics_summary[tech] = {
            "xgb_model": _compute_metrics(
                y_test, y_pred_xgb, y_test_nat, y_pred_xgb_nat, eval_cfg, sigma=None
            ),
            "ngb_model": _compute_metrics(
                y_test, y_pred_mean_ng, y_test_nat, y_pred_mean_nat_ng, eval_cfg, sigma
            ),
        }

        # --- SHAP values ---
        logger.info("Starting SHAP value computation")
        shap_dict[tech] = {}
        shap_dict[tech]["XGBoost"] = shap_analysis(X_test, features, xgb_model) 
        shap_dict[tech]["NGBoost_mean"] = shap_analysis(X_test, features, ngb_model, model_output = 0)
        shap_dict[tech]["NGBoost_std"] = shap_analysis(X_test, features, ngb_model, model_output = 1)
        logger.info("SHAP values computed")

        # --- Confidence intervals ---
        _plot_confidence_intervals(
            tech=tech,
            X_test=X_test,
            y_test_nat=y_test_nat,
            xgb_model=xgb_model,
            ngb_model=ngb_model,
            y_norm_mean=y_norm_mean,
            y_norm_std=y_norm_stdev,
            eval_dir=eval_dir,
            ci_percentile=eval_cfg.get("confidence_intervals", {}).get("percentile", 0.95),
            display_n=eval_cfg.get("confidence_intervals", {}).get("display_samples", 50),
        )

        # --- Calibration curve ---
        _plot_calibration_curves(tech, ngb_model, X_test, y_test, eval_dir)

    # Save metrics
    metrics_path = eval_dir / "metrics_test.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    logger.info("Saved test metrics to %s", metrics_path)

    # Save SHAP
    shap_path = eval_dir / f"test_shap_values.pkl"
    joblib.dump(shap_dict, shap_path)
    logger.info("Saved SHAP values to %s", shap_path)


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_true_nat: np.ndarray,
    y_pred_nat: np.ndarray,
    eval_cfg: Dict,
    sigma: Union[np.ndarray, None] = None,
) -> Dict[str, float]:
    """
    Compute evaluation metrics (optionally filtered by config) in both
    normalized and natural units.

    Args:
        y_true: Ground-truth values (normalized scale).
        y_pred: Predicted values (normalized scale).
        y_true_nat: Ground-truth values in natural (original) units.
        y_pred_nat: Predicted values in natural units.
        eval_cfg: Configuration dictionary containing metric options.
        sigma: Optional predictive standard deviation (for CRPS).

    Returns:
        Dictionary of computed metrics.
    """
    eval_metrics = eval_cfg.get("metrics", {})
    metrics: Dict[str, float] = {}

    # --- Normalized-space metrics ---
    if eval_metrics.get("mae", True):
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
    if eval_metrics.get("rmse", True):
        metrics["rmse"] = float(root_mean_squared_error(y_true, y_pred))
    if eval_metrics.get("r2", True):
        metrics["r2"] = float(r2_score(y_true, y_pred))

    # --- Natural-space metrics ---
    if eval_metrics.get("mae_nat", True):
        metrics["mae_nat"] = float(mean_absolute_error(y_true_nat, y_pred_nat))
    if eval_metrics.get("rmse_nat", True):
        metrics["rmse_nat"] = float(root_mean_squared_error(y_true_nat, y_pred_nat))

    # --- CRPS ---
    if eval_metrics.get("crps", True):
        if sigma is not None:
            metrics["crps"] = float(np.mean(_crps_gaussian(y_true, y_pred, sigma)))
        else:
            metrics["crps"] = metrics.get("mae", np.nan)

    return metrics


def _phi(z: np.ndarray) -> np.ndarray:
    """Standard normal probability density function (PDF)."""
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z**2)


def _Phi(z: np.ndarray) -> np.ndarray:
    """Standard normal cumulative distribution function (CDF)."""
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def _crps_gaussian(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Continuous Ranked Probability Score (CRPS) for Gaussian predictive distributions.

    Args:
        y: True target values.
        mu: Predicted means.
        sigma: Predicted standard deviations.
        eps: Numerical stability term for sigma (default: 1e-12).

    Returns:
        Array of CRPS values, one per sample.
    """
    sigma = np.maximum(np.asarray(sigma, dtype=float), eps).flatten()
    mu = np.asarray(mu, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    z = (y - mu) / sigma

    return sigma * (z * (2.0 * _Phi(z) - 1.0) + 2.0 * _phi(z) - 1.0 / np.sqrt(np.pi))


def _plot_confidence_intervals(
    tech: str,
    X_test: np.ndarray,
    y_test_nat: np.ndarray,
    xgb_model: XGBRegressor,
    ngb_model: NGBRegressor,
    y_norm_mean: float,
    y_norm_std: float,
    eval_dir: Path,
    ci_percentile: float = 0.95,
    display_n: int = 50,
) -> None:
    """
    Plot confidence intervals for NGBoost predictions and save to PDF.

    Args:
        tech: Radio technology name (e.g., '4G', '5G_SA').
        X_test: Test features.
        y_test_nat: Ground truth in natural units.
        xgb_model: Trained XGBoost model.
        ngb_model: Trained NGBoost model.
        y_norm_mean: Mean of normalized target.
        y_norm_std: Standard deviation of normalized target.
        eval_dir: Directory for saving evaluation plots.
        ci_percentile: Confidence interval percentile (default: 0.95).
        display_n: Number of samples to visualize (default: 50).
    """
    z = scipy.stats.norm.ppf(0.5 + ci_percentile / 2)
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_dist = ngb_model.pred_dist(X_test)
    mu = y_pred_dist.mean()
    std = y_pred_dist.std()

    # Convert back to natural scale
    y_pred_xgb_nat = np.expm1(y_pred_xgb * y_norm_std + y_norm_mean)
    y_pred_mu_nat = np.expm1(mu * y_norm_std + y_norm_mean)
    lb = np.expm1((mu - z * std) * y_norm_std + y_norm_mean)
    ub = np.expm1((mu + z * std) * y_norm_std + y_norm_mean)

    idx = np.random.choice(len(X_test), min(display_n, len(X_test)), replace=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, j in enumerate(idx):
        lower_i, upper_i, y_true = lb[j], ub[j], y_test_nat[j]
        inside = lower_i <= y_true <= upper_i
        color = "green" if inside else "red"
        ax.plot([i, i], [lower_i, upper_i], color=color, linewidth=8, alpha=0.4)
        ax.plot(i, y_true, "x", color="black", label="Measured" if i == 0 else "")
        ax.plot(i, y_pred_xgb_nat[j], "o", color="orange", label="XGBoost" if i == 0 else "")
        ax.plot(i, y_pred_mu_nat[j], "o", color="blue", label=r"NGBoost $\mu$" if i == 0 else "")

    sigma_handle = mlines.Line2D([], [], color="red", linewidth=6, alpha=0.4)
    sigma_handle2 = mlines.Line2D([], [], color="green", linewidth=6, alpha=0.4)
    handles, labels = ax.get_legend_handles_labels()
    handles.append((sigma_handle, sigma_handle2))
    labels.append(r"NGBoost $\sigma$")

    ax.legend(handles, labels, fontsize=14, handler_map={tuple: HandlerTuple(ndivide=None)})
    ax.set_xlabel("Sample index", fontsize=16)
    ax.set_ylabel("Throughput (Kbps)", fontsize=16)
    ax.set_yscale("log")
    ax.grid(True)
    plt.tight_layout()

    fig_path = eval_dir / f"{tech}_CIs.pdf"
    fig.savefig(fig_path, format="pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    logger.info("Saved confidence interval plot to %s", fig_path)


def _plot_calibration_curves(
    tech: str,
    ngb_model: NGBRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    eval_dir: Path,
    alphas: np.ndarray = np.linspace(0.01, 0.99, 25),
) -> None:
    """
    Plot calibration curves for NGBoost predictions.

    Args:
        tech: Radio technology name.
        ngb_model: Trained NGBoost model.
        X_test: Test features.
        y_test: True normalized targets.
        eval_dir: Directory for saving evaluation plots.
        alphas: Confidence levels to evaluate (default: 0.01–0.99).
    """
    dist_test = ngb_model.pred_dist(X_test)
    means = dist_test.mean().flatten()
    stds = dist_test.std().flatten()

    coverage, interval_length = [], []
    for alpha in alphas:
        z = scipy.stats.norm.ppf(0.5 + alpha / 2)
        lower = means - z * stds
        upper = means + z * stds
        inside = (y_test >= lower) & (y_test <= upper)
        coverage.append(np.mean(inside))
        interval_length.append((upper - lower).mean())

    diff = np.array(coverage) - alphas
    ece = np.trapz(np.abs(diff), alphas)

    plt.figure(figsize=(8, 6))
    plt.plot(alphas, coverage, "X--", label=f"{tech} (C-AUC={ece:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Ideal")
    plt.xlabel("Nominal Confidence Level (1 - α)", fontsize=14)
    plt.ylabel("Empirical Coverage", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()

    fig_path = eval_dir / f"{tech}_calibration.pdf"
    plt.savefig(fig_path, format="pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close()
    logger.info("Saved calibration curve to %s", fig_path)


def shap_analysis(X_shap: np.ndarray, features: list[str], model: Any, model_output: int = 0) -> Any:
    """
    Compute and save SHAP values and beeswarm plots for a trained model.

    Args:
        X_shap (np.ndarray): Feature matrix for SHAP computation.
        features (list[str]): List of feature names.
        model (Any): Trained model (e.g., XGBRegressor or NGBRegressor).
        model_output (int, optional): Model output index for SHAP (default=0).

    Returns:
        Any: Computed SHAP values.
    """

    model_class = type(model).__name__
    model_name = "NGBoost" if "NGB" in model_class else "XGBoost" if "XGB" in model_class else model_class

    if model_name == "XGBoost":
        explainer = shap.TreeExplainer(
            model, feature_names=features
        )
        shap_values = explainer(X_shap)
    elif model_name == "NGBoost":
        explainer = shap.TreeExplainer(
            model, model_output = model_output, feature_names=features
        )
        shap_values = explainer(X_shap)

    return shap_values
