"""Utility functions for logging setup and configuration loading."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Union, Optional, List

import shutil
import yaml

def setup_logger(config: Dict) -> None:
    """
    Configure the logging system based on the provided configuration dictionary.

    Args:
        config (dict): Configuration dictionary loaded from a YAML file.
            Expected to include a "logging" section with optional keys:
                - level (str): Logging level (e.g., "INFO", "DEBUG").
                - format (str): Message format string.
                - datefmt (str): Datetime format for log entries.
    """
    log_cfg = config.get("logging", {})

    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO),
        format=log_cfg.get(
            "format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ),
        datefmt=log_cfg.get("datefmt", "%H:%M:%S"),
        force=True,  # reconfigure logging (important in Jupyter)
    )

    logging.getLogger().info(
        "Logger initialized (level=%s)", log_cfg.get("level", "INFO").upper()
    )


def update_file_logger(config: Dict, exp_dir: Optional[Path]) -> Path:
    """
    Add file logging to the existing logger once the experiment directory is known.

    Args:
        config (Dict): Configuration dictionary, may include logging level.
        exp_dir (Path | None): Path to the experiment directory.

    Returns:
        Path: Path to the log file created.
    """
    log_dir = Path(exp_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    log_level = config.get("logging", {}).get("level", "INFO").upper()

    # Remove old handlers before re-adding (avoids duplicate console/file output)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Set up combined console + file logging
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info("Full logging initialized. Output file: %s", log_file)

    return log_file


def save_config_copy(config_path: str, exp_dir: Path) -> Path:
    """
    Save a copy of the configuration YAML file inside the experiment directory.

    Args:
        config_path (str): Path to the original YAML configuration file.
        exp_dir (Path): Path to the experiment directory.

    Returns:
        Path: Path to the saved configuration file inside the experiment folder.
    """
    logger = logging.getLogger(__name__)
    config_dst = exp_dir / "config.yaml"

    try:
        shutil.copy2(config_path, config_dst)
        logger.info("Copied configuration file to: %s", config_dst)
    except Exception as e:
        logger.warning("Failed to copy config file: %s", e)

    return config_dst


def load_config(config_path: Union[str, Path] = "configs/experiment.yaml") -> Dict:
    """
    Load the experiment configuration from a YAML file.

    Args:
        config_path (str | Path): Path to the YAML configuration file.

    Returns:
        dict: Loaded configuration dictionary.
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading configuration from %s", config_path)

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    logger.debug("Configuration loaded with keys: %s", list(config.keys()))
    return config


def create_experiment_dir(config: Dict, experiment_name: Optional[str] = None) -> Path:
    """
    Create a timestamped experiment directory and return its path.

    The directory is created inside the base artifacts folder with the format:
    `YYYY-MM-DD_HH-MM-SS[_experiment_name]`.

    Subdirectories for models, data splits, logs, and evaluation results
    are also initialized automatically.

    Args:
        config (dict): Configuration dictionary expected to contain:
            experiment.output_dir (optional): Base directory for artifacts.
        experiment_name (Optional[str]): Optional descriptive name to append
            to the experiment folder (e.g., "spain_5g_test").

    Returns:
        Path: Path object pointing to the created experiment directory.
    """
    logger = logging.getLogger(__name__)

    # Base directory (defaults to "artifacts/")
    base_dir = config.get("experiment", {}).get("base_dir", "artifacts")
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Base directory: %s", base_dir)

    # Timestamp + optional experiment name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{timestamp}_{experiment_name}" if experiment_name else timestamp

    exp_dir = base_dir / folder_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create standard subfolders
    for subfolder in ["models", "data_splits", "logs", "eval"]:
        (exp_dir / subfolder).mkdir(exist_ok=True)

    logger.info("Experiment directory created: %s", exp_dir)
    return exp_dir


def get_latest_experiment_dir(artifacts_dir: Union[str, Path] = "artifacts") -> Path:
    """
    Retrieve the most recently created experiment directory.

    Scans the given artifacts directory and returns the folder
    with the most recent modification timestamp.

    Args:
        artifacts_dir (Union[str, Path]): Path to the base artifacts directory.
            Can be provided as a string or Path. Defaults to "artifacts".

    Returns:
        Path: Path to the most recently modified experiment directory.

    Raises:
        FileNotFoundError: If the directory doesn't exist or has no subdirectories.
    """
    logger = logging.getLogger(__name__)
    artifacts_dir = Path(artifacts_dir) 

    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {artifacts_dir}")

    # Gather subdirectories (ignoring non-dirs)
    exp_dirs = [d for d in artifacts_dir.iterdir() if d.is_dir()]
    if not exp_dirs:
        raise FileNotFoundError(f"No experiment directories found in {artifacts_dir}")

    # Pick the most recently modified
    latest_dir = max(exp_dirs, key=lambda d: d.stat().st_mtime)
    logger.info("Latest experiment directory found: %s", latest_dir)

    return latest_dir


def get_project_root() -> Path:
    """
    Return the absolute path to the project root directory.
    Works both inside notebooks and regular Python modules.
    """
    current = Path(__file__).resolve() if "__file__" in globals() else Path().resolve()
    for parent in current.parents:
        if (parent / "configs").exists() and (parent / "src").exists():
            return parent
    raise RuntimeError("Project root not found — make sure you're inside the repo structure.")


def categorize_features(features: List[str]) -> Dict[str, List[str]]:
    """
    Categorize a list of feature names into predefined groups.

    Args:
        features (List[str]): List of feature names to categorize.

    Returns:
        Dict[str, List[str]]: Dictionary mapping category names 
        ("deployment", "radio", "e2e", "context") to lists of matching features.
    """
    categories = {
        "deployment": ["Frequency Band", "Carrier"],
        "radio": ["RSRP", "RSRQ", "SINR", "Timing Advance"],
        "e2e": ["Latency", "Jitter", "Packet Loss", "DL TTFB"],
        "context": ["Time of the Day", "Day of the Week"],
    }

    return {
        name: [f for f in features if f in group]
        for name, group in categories.items()
    }
