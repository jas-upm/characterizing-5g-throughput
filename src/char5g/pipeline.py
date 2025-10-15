"""Main pipeline for 5G throughput modeling and evaluation."""

import logging
from pathlib import Path
from typing import Optional

from src.char5g.utils import (
    load_config,
    setup_logger,
    update_file_logger,
    create_experiment_dir,
    get_latest_experiment_dir,
    save_config_copy
)
from src.char5g.data_loader import get_data
from src.char5g.train import train_models
from src.char5g.eval import evaluate_models

logger = logging.getLogger(__name__)


def resolve_experiment_dir(
    config: dict,
    exp_dir: Optional[str],
    resume: bool,
    experiment_name: Optional[str],
) -> Path:
    """
    Determine which experiment directory to use.

    Logic:
        - If ``exp_dir`` is provided, use it (must exist).
        - If ``resume`` is True, load the most recent experiment directory.
        - Otherwise, create a new timestamped directory (optionally named).

    Args:
        config (dict): Configuration dictionary.
        exp_dir (Optional[str]): Explicit experiment directory path.
        resume (bool): Whether to resume from the most recent experiment.
        experiment_name (Optional[str]): Optional experiment name suffix.

    Returns:
        Path: Path to the resolved experiment directory.
    """
    if exp_dir:
        path = Path(exp_dir)
        if not path.exists():
            raise FileNotFoundError(f"Experiment directory not found: {path}")
        logger.info("Using specified experiment directory: %s", path)
        return path

    if resume:
        artifacts_path = Path(config.get("experiment", {}).get("output_dir", "artifacts"))
        path = get_latest_experiment_dir(artifacts_path)
        logger.info("Resuming from latest experiment directory: %s", path)
        return path

    path = create_experiment_dir(config, experiment_name=experiment_name)
    logger.info("Created new experiment directory: %s", path)
    return path


def run_pipeline(
    config_path: str = "configs/experiment.yaml",
    exp_dir: Optional[str] = None,
    resume: bool = False,
    experiment_name: Optional[str] = None,
) -> None:
    """
    Execute the full 5G throughput characterization workflow.

    Steps:
        1. Load experiment configuration.
        2. Initialize console logger for startup messages.
        3. Determine or create experiment directory.
        4. Reconfigure logger to output the log to the experiment folder 
            Save copy of config YAML file.
        5. Load data
        6. Train models
        7. Evaluate on test set.

    Args:
        config_path (str): Path to the YAML configuration file.
        exp_dir (Optional[str]): Optional existing experiment directory.
        resume (bool): Whether to resume from the most recent experiment.
        experiment_name (Optional[str]): Optional name for the new experiment.
    """
    # --- 1. Load configuration ---
    config = load_config(config_path)

    # --- 2. Initialize console logger ---
    setup_logger(config)
    logger.info("Starting 5G throughput characterization pipeline...")

    # --- 3. Resolve experiment directory ---
    exp_dir_path = resolve_experiment_dir(config, exp_dir, resume, experiment_name)
    logger.info("Experiment directory resolved: %s", exp_dir_path)

    # --- 4. Update logger to include file logging ---
    log_file = update_file_logger(config, exp_dir=exp_dir_path)
    logger.info("Logging to file: %s", log_file)
    config_dst = save_config_copy(config_path, exp_dir_path)
    logger.info("Config file saved to: %s", config_dst)

    # --- 5. Load data ---
    Xy_train_val_test = get_data(exp_dir=exp_dir_path)
    logger.info("Data loaded and preprocessed successfully.")

    # --- 6. Train models ---
    train_models(Xy_train_val_test, config, exp_dir_path)
    logger.info("Model training completed successfully.")

    # --- 7. Evaluate models ---
    evaluate_models(Xy_train_val_test, config, exp_dir_path)
    logger.info("Model evaluation completed successfully.")

    logger.info("Pipeline finished successfully. Results stored in: %s", exp_dir_path)