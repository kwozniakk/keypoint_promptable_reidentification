import argparse
import torch

from torchreid.scripts.builder import build_config, build_torchreid_model_engine
from torchreid.scripts.default_config import engine_run_kwargs


def main():
    parser = argparse.ArgumentParser(
        description="Train KPR on a custom dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a training config YAML file",
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--save_dir",
        default="log/custom_train",
        help="Directory to store logs and checkpoints",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        default=None,
        help="Override config options using the command line",
    )
    args = parser.parse_args()

    cfg = build_config(args=args, config_path=args.config)
    cfg.use_gpu = torch.cuda.is_available()
    cfg.data.root = args.root
    cfg.data.save_dir = args.save_dir

    engine, _ = build_torchreid_model_engine(cfg)
    engine.run(**engine_run_kwargs(cfg))


if __name__ == "__main__":
    main()
