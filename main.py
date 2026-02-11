"""Entry-point for diffusion inference/evaluation."""

import argparse
import os
import random
import warnings
import yaml

import dotenv
import ml_collections
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import wandb


dotenv.load_dotenv(".env")

from src.constants import DATA_PATH
from src.data.dataset import BatchSampler, RNADesignDataset
from src.model import GVPDiff
from src.diffusion import (
    build_noise_scheduler,
    evaluate_dataset_best_of_n,
    evaluate_single_sample_best_of_n,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_CLI_OVERRIDE_KEYS = (
    "test_id",
    "train_id",
    "sample_repeats",
    "temperature",
    "test_pdb_id",
    "train_pdb_id",
)


def _torch_load(path: str, map_location=None, weights_only: bool = False):
    """Load checkpoint from disk."""
    return torch.load(path, map_location=map_location, weights_only=weights_only)


def _load_config(path: str) -> ml_collections.ConfigDict:
    """Load config YAML with wandb-style {'value': ...} entries."""
    with open(path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    config_dict = {}
    for key, value in raw_cfg.items():
        if isinstance(value, dict) and "value" in value:
            config_dict[key] = value["value"]
        else:
            config_dict[key] = value
    return ml_collections.ConfigDict(config_dict)


def _apply_cli_overrides(config: ml_collections.ConfigDict, args: argparse.Namespace) -> None:
    """Apply only explicitly provided CLI overrides to config."""
    for key in _CLI_OVERRIDE_KEYS:
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    if args.eval_all_test:
        config.eval_all_test = True
    if args.deterministic:
        config.deterministic = True


def _load_checkpoint(model: torch.nn.Module, path, device: torch.device) -> None:
    state = _torch_load(path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state_dict = state["model"]
    else:
        state_dict = state

    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}

    model_state = model.state_dict()
    loadable_state = {}
    skipped_shape = []
    unexpected_keys = []
    for key, value in state_dict.items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        loadable_state[key] = value

    missing_keys = [key for key in model_state.keys() if key not in loadable_state]
    model.load_state_dict(loadable_state, strict=False)


def main(config: ml_collections.ConfigDict, device: torch.device) -> None:

    set_seed(config.seed)

    model = build_model(config).to(device)
    total_param = sum(np.prod(p.size()) for p in model.parameters())
    wandb.run.summary["total_param"] = total_param

    if not config.model_path:
        raise ValueError("Inference mode requires `model_path` in config.")
    _load_checkpoint(model, config.model_path, device)

    eval_mode = _resolve_eval_mode(config)
    train_loader, test_loader = build_dataloaders(
        config,
        need_train=(eval_mode in ("single_train", "single_train_pdb")),
        need_test=(eval_mode in ("single_test", "single_test_pdb", "all_test")),
        train_sample_index=(int(config.train_id) if eval_mode == "single_train" else None),
        train_pdb_id=(str(config.train_pdb_id) if eval_mode == "single_train_pdb" else None),
        test_sample_index=(int(config.test_id) if eval_mode == "single_test" else None),
        test_pdb_id=(str(config.test_pdb_id) if eval_mode == "single_test_pdb" else None),
    )
    print("Starting inference.")
    noise_scheduler = build_noise_scheduler(config)

    if eval_mode == "all_test":
        result = evaluate_dataset_best_of_n(
            model=model,
            dataset=test_loader.dataset,
            noise_scheduler=noise_scheduler,
            n_steps=config.n_steps,
            device=device,
            n_repeats=config.sample_repeats,
            temperature=config.temperature,
            deterministic=config.deterministic,
        )
        print(f"Average Test Recovery: {result['avg_best_acc']:.4f}")
        wandb.log({"test/avg_best_acc": result["avg_best_acc"]})
        return

    if eval_mode == "single_train":
        selected_split, selected_id, selected_dataset = "train", int(config.train_id), train_loader.dataset
    elif eval_mode == "single_train_pdb":
        selected_split, selected_id, selected_dataset = "train", str(config.train_pdb_id), train_loader.dataset
    elif eval_mode == "single_test":
        selected_split, selected_id, selected_dataset = "test", int(config.test_id), test_loader.dataset
    elif eval_mode == "single_test_pdb":
        selected_split, selected_id, selected_dataset = "test", str(config.test_pdb_id), test_loader.dataset
    else:
        raise ValueError(
            "Set one of `train_id`, `train_pdb_id`, `test_id`, `test_pdb_id`, "
            "or set `eval_all_test=true`."
        )

    if len(selected_dataset.data_list) == 0:
        raise ValueError(
            f"No valid sample for {selected_split}_id={selected_id}. "
        )

    # Single-sample mode now preprocesses only one sample from the requested split index.
    selected_raw_data = selected_dataset.data_list[0]

    result = evaluate_single_sample_best_of_n(
        model=model,
        dataset=selected_dataset,
        raw_data=selected_raw_data,
        noise_scheduler=noise_scheduler,
        n_steps=config.n_steps,
        device=device,
        n_repeats=config.sample_repeats,
        temperature=config.temperature,
        deterministic=config.deterministic,
    )

    print(
        f"{selected_split}_id={selected_id} "
        f"Recovery: {result['best_acc']:.4f}"
    )
    print(f"Designed sequence: {result['designed_seq']}")
    wandb.log({f"{selected_split}/best_acc": result["best_acc"]})


def build_dataloaders(
    config: ml_collections.ConfigDict,
    need_train: bool = True,
    need_test: bool = True,
    train_sample_index: int = None,
    train_pdb_id: str = None,
    test_sample_index: int = None,
    test_pdb_id: str = None,
):
    """Construct train/test dataloaders for requested splits only."""

    train_list, _, test_list = get_data_splits(
        config.split,
        train_sample_index=train_sample_index,
        train_pdb_id=train_pdb_id,
        test_sample_index=test_sample_index,
        test_pdb_id=test_pdb_id,
    )

    train_loader = None
    test_loader = None

    if need_train:
        trainset = get_dataset(config, train_list, split="train")
        train_loader = get_dataloader(config, trainset, shuffle=True)
    if need_test:
        testset = get_dataset(config, test_list, split="test")
        test_loader = get_dataloader(config, testset, shuffle=False)

    return train_loader, test_loader


def _resolve_eval_mode(config: ml_collections.ConfigDict) -> str:
    if bool(config.eval_all_test):
        return "all_test"

    has_train_id = int(config.train_id) >= 0
    has_test_id = int(config.test_id) >= 0
    has_train_pdb = bool(str(config.get("train_pdb_id", "")).strip())
    has_test_pdb = bool(str(config.get("test_pdb_id", "")).strip())

    if has_train_id and has_train_pdb:
        raise ValueError("Only one of `train_id` or `train_pdb_id` can be set.")
    if has_test_id and has_test_pdb:
        raise ValueError("Only one of `test_id` or `test_pdb_id` can be set.")

    use_train = has_train_id or has_train_pdb
    use_test = has_test_id or has_test_pdb
    if use_train and use_test:
        raise ValueError("Only one of train-mode or test-mode selectors can be set.")

    if has_train_id:
        return "single_train"
    if has_train_pdb:
        return "single_train_pdb"
    if has_test_pdb:
        return "single_test_pdb"
    if has_test_id:
        return "single_test"

    raise ValueError(
        "Set one of `train_id`, `train_pdb_id`, `test_id`, `test_pdb_id`, "
        "or set `eval_all_test=true`."
    )


def get_data_splits(
    split_type: str = "structsim_v2",
    train_sample_index: int = None,
    train_pdb_id: str = None,
    test_sample_index: int = None,
    test_pdb_id: str = None,
):
    """Return train/val/test splits as raw data lists."""

    data_list = list(_torch_load(os.path.join(DATA_PATH, "processed.pt"), weights_only=False).values())

    def select_indices(indices, sample_index=None, pdb_id=None, split_name=""):
        if sample_index is None:
            if pdb_id is None:
                return [data_list[index] for index in indices]
            for local_idx, global_idx in enumerate(indices):
                sample = data_list[global_idx]
                id_list = sample.get("id_list", [])
                if pdb_id in id_list:
                    conf_idx = id_list.index(pdb_id)
                    coords_list = sample.get("coords_list", [])
                    if conf_idx >= len(coords_list):
                        raise ValueError(
                            f"Matched pdb id `{pdb_id}` at conf index {conf_idx}, "
                            f"but coords_list has length {len(coords_list)}."
                        )

                    # Keep only the selected conformation so downstream fixed_idx=0 is correct.
                    selected_sample = dict(sample)
                    selected_sample["coords_list"] = [coords_list[conf_idx]]
                    selected_sample["id_list"] = [pdb_id]

                    print(
                        f"Matched {split_name}_pdb_id={pdb_id} to {split_name}_id={local_idx}, "
                        f"conformer_idx={conf_idx}"
                    )
                    return [selected_sample]
            raise ValueError(f"{split_name}_pdb_id `{pdb_id}` not found in split `{split_type}` {split_name} set.")
        if sample_index < 0 or sample_index >= len(indices):
            raise IndexError(f"sample index {sample_index} out of range (size={len(indices)})")
        return [data_list[indices[sample_index]]]

    train_idx_list, val_idx_list, test_idx_list = _torch_load(
        os.path.join(DATA_PATH, f"{split_type}_split.pt"),
        weights_only=False,
    )
    return (
        select_indices(train_idx_list, train_sample_index, train_pdb_id, "train"),
        select_indices(val_idx_list),
        select_indices(test_idx_list, test_sample_index, test_pdb_id, "test"),
    )


def get_dataset(config: ml_collections.ConfigDict, data_list, split: str = "train"):
    """Instantiate an :class:`RNADesignDataset` for the provided split."""

    return RNADesignDataset(
        data_list=data_list,
        split=split,
        radius=config.radius,
        top_k=config.top_k,
        num_rbf=config.num_rbf,
        num_posenc=config.num_posenc,
        max_num_conformers=config.max_num_conformers,
        noise_scale=config.noise_scale,
    )


def get_dataloader(
    config: ml_collections.ConfigDict,
    dataset: RNADesignDataset,
    shuffle: bool = True,
    pin_memory: bool = True,
):
    """Return a :class:`torch_geometric.loader.DataLoader` for the dataset."""

    sampler = BatchSampler(
        node_counts=dataset.node_counts,
        max_nodes_batch=config.max_nodes_batch,
        max_nodes_sample=config.max_nodes_sample,
        shuffle=shuffle,
    )
    return DataLoader(
        dataset,
        num_workers=config.num_workers,
        batch_sampler=sampler,
        pin_memory=pin_memory,
    )


def build_model(config: ml_collections.ConfigDict) -> torch.nn.Module:
    """Create the diffusion model architecture."""
    return GVPDiff(config)


def set_seed(seed: int = 0) -> None:
    """Ensure deterministic behaviour across Python, NumPy and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", default="configs/default.yaml", type=str)
    parser.add_argument("--expt_name", dest="expt_name", default=None, type=str)
    parser.add_argument("--tags", nargs="+", dest="tags", default=[])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--test_id", dest="test_id", default=None, type=int)
    parser.add_argument("--train_id", dest="train_id", default=None, type=int)
    parser.add_argument("--sample_repeats", dest="sample_repeats", default=None, type=int)
    parser.add_argument("--temperature", dest="temperature", default=None, type=float)
    parser.add_argument("--train_pdb_id", dest="train_pdb_id", default=None, type=str)
    parser.add_argument("--test_pdb_id", dest="test_pdb_id", default=None, type=str)
    parser.add_argument("--eval_all_test", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args, _ = parser.parse_known_args()

    config = _load_config(args.config)
    _apply_cli_overrides(config, args)

    use_wandb_cfg = bool(config.get("use_wandb", False))
    use_wandb = (use_wandb_cfg or args.use_wandb) and not args.no_wandb
    wandb_mode = "online" if use_wandb else "disabled"
    wandb.init(
        project=os.environ.get("WANDB_PROJECT"),
        entity=os.environ.get("WANDB_ENTITY"),
        config=dict(config),
        name=args.expt_name,
        tags=args.tags,
        mode=wandb_mode,
    )
    config.use_wandb = bool(use_wandb)

    device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")

    main(config, device)
