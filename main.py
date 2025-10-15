"""Command-line entry point for running the 5G throughput modeling pipeline."""

import argparse

from src.char5g.pipeline import run_pipeline


def main() -> None:
    """
    Parse command-line arguments and execute the pipeline.

    Command-line flags:
        --config           Path to the configuration YAML file.
        --exp_dir          Use an existing experiment directory instead of creating a new one.
        --resume           Resume from the most recent experiment directory.
        --experiment_name  Optional name suffix for the new experiment directory.
    """
    parser = argparse.ArgumentParser(
        description="Run the 5G throughput modeling and evaluation pipeline. Creates a new directory in /artifacts to save the results with the current timestamp as the folder name."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yaml",
        help="Path to the configuration YAML file (default: configs/experiment.yaml).",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=None,
        help="Path to an existing experiment directory to reuse.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the most recent experiment directory if no directory is specified.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Optional name suffix for the new experiment directory (e.g., 'spain_5g_test').",
    )

    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        exp_dir=args.exp_dir,
        resume=args.resume,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()