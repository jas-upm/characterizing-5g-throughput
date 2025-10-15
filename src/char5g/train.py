"""Training routines for XGBoost (baseline) and NGBoost (uncertainty-aware) models."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import shap
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from ngboost import NGBRegressor

logger = logging.getLogger(__name__)


def _train_for_tech(
    tech: str,
    Xy_train_val: Dict[str, Dict[str, Any]],
    features,
    config: Dict,
    exp_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """
    Private helper: Train all models for a single technology and return the results and metrics.

    Args:
        tech (str): Radio technology name (e.g., "4G", "5G_SA").
        Xy_train_val (dict): Data dictionary for all technologies.
        features (list[str]): Feature list from configuration.
        config (dict): Experiment configuration dictionary.
        exp_dir (Path): Base experiment directory.

    Returns:
        Tuple of:
            - dict: models and metrics for this technology.
            - dict: summary metrics for easy aggregation.
    """
    logger.info("===== Training for %s =====", tech)

    save_dir = exp_dir / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving training outputs to %s", save_dir)

    X_train = Xy_train_val[tech]["X_train"]
    y_train = Xy_train_val[tech]["y_train"]
    X_val = Xy_train_val[tech]["X_val"]
    y_val = Xy_train_val[tech]["y_val"]

    logger.info(
        "Dataset shapes — X_train: %s, X_val: %s, y_train: %s, y_val: %s",
        X_train.shape,
        X_val.shape,
        y_train.shape,
        y_val.shape,
    )

    # --- Train both models ---
    xgb_model, xgb_metrics = train_xgboost_model(
        X_train, y_train, X_val, y_val, config, save_dir, tech,
    )
    ngb_model, ngb_metrics = train_ngboost_model(
        X_train, y_train, X_val, y_val, config, save_dir, tech,
    )

    results = {
        "xgb_model": xgb_model,
        "ngb_model": ngb_model,
        "metrics": {"XGBoost": xgb_metrics, "NGBoost": ngb_metrics},
    }
    summary = {"XGBoost": xgb_metrics, "NGBoost": ngb_metrics}

    logger.info("Completed training for %s", tech)
    return results, summary


def train_models(
    Xy_train_val: Dict[str, Dict[str, Any]],
    config: Dict,
    exp_dir: Union[str, Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Public entry point: orchestrates model training across technologies.
    """
    data_cfg = config.get("data", {})
    radio_techs = data_cfg.get("radio_access_technologies", [])
    features = data_cfg.get("features", []) + data_cfg.get("temporal_encodings", [])

    results: Dict[str, Dict[str, Any]] = {}
    summary_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    exp_dir = Path(exp_dir)

    for tech in radio_techs:
        tech_results, tech_summary = _train_for_tech(tech, Xy_train_val, features, config, exp_dir)
        results[tech] = tech_results
        summary_metrics[tech] = tech_summary

    # --- Save global summary of metrics ---
    summary_path = exp_dir / "models" / "summary_metrics_val.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=4)
    logger.info("Saved summary metrics to %s", summary_path)

    return results


def train_xgboost_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
    y_val: np.ndarray, config: Dict[str, Any], save_dir: Path, tech: str) -> Tuple[Any, Dict[str, float]]:
    """Train an XGBoost regressor and compute validation metrics.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation targets.
        config (Dict[str, Any]): Configuration dictionary.
        save_dir (Path): Directory where the model will be saved.
        tech (str): Name of the radio technology (e.g., '4G', '5G_NSA').

    Returns:
        Tuple[Any, Dict[str, float]]: The trained XGBoost model and validation metrics.
    """
    
    """Train an XGBoost regressor and compute validation metrics."""

    model_cfg = config.get("models", {}).get("XGBoost", {})
    experiment_cfg = config.get("experiment", {})
    params = {
        "n_estimators": model_cfg.get("n_estimators", 1000),
        "learning_rate": model_cfg.get("learning_rate", 0.05),
        "eval_metric": ["rmse", "mae"],
        "early_stopping_rounds": model_cfg.get("patience", 100),
        "random_state": experiment_cfg.get("random_state", 42),
    }

    logger.info("Initializing XGBoost with params: %s", params)
    model = XGBRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    y_pred = model.predict(X_val)
    metrics = {
        "MAE": float(mean_absolute_error(y_val, y_pred)),
        "MSE": float(mean_squared_error(y_val, y_pred)),
    }
    logger.info("XGBoost validation metrics: %s", metrics)

    # --- Save model ---
    model_path = save_dir / f"XGBoost_model_{tech}.pkl"
    joblib.dump(model, model_path)
    logger.info("Saved XGBoost model to %s", model_path)

    return model, metrics


def train_ngboost_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
    y_val: np.ndarray, config: Dict[str, Any], save_dir: Path, tech: str) -> Tuple[Any, Dict[str, float]]:
    """Train an NGBoost regressor and compute validation metrics.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation targets.
        config (Dict[str, Any]): Configuration dictionary.
        save_dir (Path): Directory where the model will be saved.
        tech (str): Name of the radio technology (e.g., '4G', '5G_NSA').

    Returns:
        Tuple[Any, Dict[str, float]]: The trained NGBoost model and validation metrics.
    """

    model_cfg = config.get("models", {}).get("NGBoost", {})
    experiment_cfg = config.get("experiment", {})
    params = {
        "n_estimators": model_cfg.get("n_estimators", 1000),
        "learning_rate": model_cfg.get("learning_rate", 0.05),
        "early_stopping_rounds": model_cfg.get("patience", 100),
        "random_state": experiment_cfg.get("random_state", 42),
    }

    logger.info("Initializing NGBoost with params: %s", params)
    model = NGBRegressor(**params)
    model.fit(X_train, y_train, X_val=X_val, Y_val=y_val)

    y_pred = model.pred_dist(X_val).loc
    metrics = {
        "MAE": float(mean_absolute_error(y_val, y_pred)),
        "MSE": float(mean_squared_error(y_val, y_pred)),
    }
    logger.info("NGBoost validation metrics: %s", metrics)

    # --- Save model ---
    model_path = save_dir / f"NGBoost_model_{tech}.pkl"
    joblib.dump(model, model_path)
    logger.info("Saved NGBoost model to %s", model_path)

    return model, metrics