"""Data loading and preprocessing for the 5G throughput characterization project."""

import logging
from pathlib import Path
from typing import Tuple, Dict, Union

import numpy as np
import pandas as pd
import joblib

from src.char5g.utils import load_config

logger = logging.getLogger(__name__)

def load_data(config: Dict) -> pd.DataFrame:
    """
    Load the dataset specified in the configuration.

    Args:
        config (dict): Experiment configuration dictionary.

    Returns:
        pd.DataFrame: Raw dataset.
    """
    data_path = Path(config.get("data", {}).get("input_csv", "data/throughput.csv"))
    logger.info(f"Loading dataset from {data_path}")

    if not data_path.exists():
        logger.error(f"Dataset not found at {data_path}")
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Dataset loaded successfully with shape {df.shape}")
    return df


def preprocess_data(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Basic preprocessing.

    Filters data to include desired carriers, technologies, and features.
    Converts timestamps, adds temporal encodings, and encodes categorical features.
    """
    logger.info("Starting preprocessing")
    df = df.copy()

    data_cfg = config.get("data", {})

    # Filter to desired carriers (if defined)
    carriers = data_cfg.get("Carrier")
    if carriers:
        before = len(df)
        df = df.loc[df["Carrier"].isin(carriers)]
        logger.debug(f"Filtered carriers: {carriers} (remaining {len(df)}/{before})")

    # Filter to desired radio access technologies
    radio_techs = data_cfg.get("radio_access_technologies")
    if radio_techs:
        before = len(df)
        df = df.loc[df["RAT"].isin(radio_techs)]
        logger.debug(f"Filtered RATs: {radio_techs} (remaining {len(df)}/{before})")

    # Select relevant columns
    features = data_cfg.get("features", [])
    target = data_cfg.get("target", "DL Throughput")
    df = df[["Timestamp", "RAT"] + features + [target]]

    # Ensure timestamps are datetime
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Temporal encodings (safe defaults)
    temporal_encodings = data_cfg.get("temporal_encodings", [])
    if "Time of the Day" in temporal_encodings:
        df["Time of the Day"] = (
            df["Timestamp"].dt.hour
            + df["Timestamp"].dt.minute / 60
            + df["Timestamp"].dt.second / 3600
        )
        logger.debug("Added temporal feature: Time of the Day")
    if "Day of the Week" in temporal_encodings:
        df["Day of the Week"] = df["Timestamp"].dt.weekday.astype("Int64")
        logger.debug("Added temporal feature: Day of the Week")

    # Encode categorical carrier
    if "Carrier" in features and "Carrier" in df.columns:
        df["Carrier"] = df["Carrier"].astype("category").cat.codes
        logger.debug("Encoded categorical feature: Carrier")

    # Transform target variable
    if data_cfg.get("log_transform_target", False):
        df[target] = np.log1p(df[target])

    logger.info(f"Preprocessing completed. Final shape: {df.shape}")
    return df


def split_data(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training, validation, and test sets.

    Args:
        df (pd.DataFrame): Preprocessed dataset.
        config (dict): Configuration containing date boundaries.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train_df, val_df, test_df.
    """
    logger.info("Splitting data into train, validation, and test sets")
    dates = config.get("experiment", {}).get("data_split_dates", {})

    def _get_range(split_name: str):
        split = dates.get(split_name, {})
        start = split.get("start")
        end = split.get("end")
        if not start or not end:
            logger.warning(f"Missing start/end for split '{split_name}', returning empty DataFrame.")
            return df.iloc[0:0]
        subset = df.loc[(df["Timestamp"] >= start) & (df["Timestamp"] < end)]
        logger.debug(f"{split_name} split: {len(subset)} rows ({start} to {end})")
        return subset

    train_df = _get_range("train")
    val_df = _get_range("val")
    test_df = _get_range("test")

    logger.info(
        f"Split complete — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )
    return train_df, val_df, test_df


def prepare_feature_target_arrays(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                                  test_df: pd.DataFrame, config: Dict, 
                                  exp_dir: Union[str, Path] = None) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Prepare and normalize data for each radio technology.
    
    Args:
        train_df (pd.DataFrame): Preprocessed training dataset.
        val_df (pd.DataFrame): Preprocessed validation dataset.
        test_df (pd.DataFrame): Preprocessed test dataset.
        config (dict): Configuration containing date boundaries.
        exp_dir (str or Path): If passed, the output dict is saved into 
            "exp_dir/data_splits" folder as a .pkl file.
    
    Returns:
        dict[str, dict[str, np.ndarray]]: e.g.
            {
                "4G": {
                    "X_train": ...,
                    "y_train": ...,
                    "X_val": ...,
                    "y_val": ...,
                    "X_test": ...,
                    "y_test": ...,
                    "y_norm_mean": ...,
                    "y_norm_stdev": ...
                },
                "5G_NSA": {...},
                "5G_SA": {...}
            }
    """
    data_cfg = config.get("data", {})
    radio_techs = data_cfg.get("radio_access_technologies")
    target = data_cfg.get("target", "DL Throughput")
    features = data_cfg.get("features", []) + data_cfg.get("temporal_encodings", [])

    Xy_train_val_test = {}

    for technology in radio_techs:
        logger.info(f"Preparing data for technology: {technology}")

        train_df_tech = train_df.loc[train_df.RAT == technology].copy()
        val_df_tech = val_df.loc[val_df.RAT == technology].copy()
        test_df_tech = test_df.loc[test_df.RAT == technology].copy()

        train_df_tech = train_df_tech[features + [target]]
        val_df_tech = val_df_tech[features + [target]]
        test_df_tech = test_df_tech[features + [target]]

        logger.info("Radio Access Technology filtered")

        # Standardize target using train statistics
        if data_cfg.get("norm_target", False):
            y_norm_mean = train_df_tech[target].mean()
            y_norm_stdev = train_df_tech[target].std()
            for subset in [train_df_tech, val_df_tech, test_df_tech]:
                subset[target] = (subset[target] - y_norm_mean) / y_norm_stdev
            logger.info("Target variable normalized")

        # Prepare numpy arrays
        X_train = train_df_tech[features].values.astype("float32")
        y_train = train_df_tech[target].values.astype("float32").reshape(-1, 1)
        X_val = val_df_tech[features].values.astype("float32")
        y_val = val_df_tech[target].values.astype("float32").reshape(-1, 1)
        X_test = test_df_tech[features].values.astype("float32")
        y_test = test_df_tech[target].values.astype("float32").reshape(-1, 1)

        # Fill in output dict with all data
        # TODO: Reduce memory footprint by returning a single technology at a time
        Xy_train_val_test[technology] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "y_norm_mean": y_norm_mean,
            "y_norm_stdev": y_norm_stdev,
        }

        # Save preprocessed data
        if exp_dir:
            data_dir = exp_dir / "data_splits"
            data_dir.mkdir(exist_ok=True)

            file_path = data_dir / "Xy_train_val_test.pkl"
            joblib.dump(Xy_train_val_test, file_path)

            logger.info("Saved Xy_train_val_test dictionary to %s", file_path)

    return Xy_train_val_test


def get_data(config_path: str = "configs/experiment.yaml", 
             exp_dir: Union[str, Path] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Orchestrator function: load configuration, read data, preprocess, and split.

    Args:
        config_path (str): Path to the YAML configuration.
        exp_dir (str or Path): If passed, the output dict is saved into 
            "exp_dir/data_splits" folder as a .pkl file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train_df, val_df, test_df.
    """
    logger.info("Executing data loader script")
    config = load_config(config_path)
    df = load_data(config)
    df = preprocess_data(df, config)
    train_df, val_df, test_df = split_data(df, config)
    feature_target_dict = prepare_feature_target_arrays(train_df, val_df, test_df, config, exp_dir)
    logger.info("Data data loader script complete")

    return feature_target_dict