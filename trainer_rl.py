"""Reinforcement learning fine-tuning loop for the diffusion model."""

import argparse
import math
import os
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List

import dotenv
import ml_collections
import numpy as np
import torch
import torch.nn.utils as utils
from torch.optim import Adam
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm import tqdm

dotenv.load_dotenv(".env")

from src.constants import DATA_PATH, PROJECT_PATH
from src.data.dataset import RNADesignDataset
from src.evaluator_rl import evaluate
from src.model import GVPDiff
from src.noise_schedule import NoiseScheduleVP
from src.diffusion import ddim_sample_with_logprob
from tools.rhofold.config import rhofold_config
from tools.rhofold.rf import RhoFold

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = get_logger(__name__)


def _checkpoint_dir() -> str:
    project_root = os.path.dirname(os.path.abspath(__file__))
    run_id = getattr(wandb.run, "id", "local")
    out_dir = os.path.join(project_root, "outputs", "checkpoints", run_id)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _load_model_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("state_dict")
            or checkpoint.get("model_state_dict")
            or checkpoint.get("model")
            or checkpoint
        )
    else:
        state_dict = checkpoint

    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

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

    logger.info(
        "Checkpoint loaded from %s: matched=%d, skipped_shape=%d, unexpected=%d, missing=%d",
        checkpoint_path,
        len(loadable_state),
        len(skipped_shape),
        len(unexpected_keys),
        len(missing_keys),
    )
    if skipped_shape:
        logger.warning("Skipped shape-mismatched keys (first 10): %s", skipped_shape[:10])
    if unexpected_keys:
        logger.warning("Unexpected checkpoint keys (first 10): %s", unexpected_keys[:10])


def _update_baseline(previous: float, mean_reward: float, epoch: int, beta: float) -> float:
    """EMA-style baseline update used to stabilize policy gradient."""
    if epoch == 0 or mean_reward > previous:
        return mean_reward
    return beta * previous + (1 - beta) * mean_reward


@dataclass
class TrajectorySample:
    """Container for RL sample trajectories."""

    log_probs_traj: List[torch.Tensor]
    latents_traj: List[torch.Tensor]
    reward: float
    raw_data: dict
    advantage: float = 0.0

    def set_advantage(self, baseline: float) -> None:
        self.advantage = self.reward - baseline


def get_data_splits(split_type: str = "structsim_v2"):
    """Return train/val/test raw data lists for the requested split."""

    data_list = list(torch.load(os.path.join(DATA_PATH, "processed.pt"), weights_only=False).values())

    def index_list_by_indices(lst, indices):
        return [lst[index] for index in indices]

    train_idx_list, val_idx_list, test_idx_list = torch.load(
        os.path.join(DATA_PATH, f"{split_type}_split.pt"),
        weights_only=False,
    )
    train_list = index_list_by_indices(data_list, train_idx_list)
    val_list = index_list_by_indices(data_list, val_idx_list)
    test_list = index_list_by_indices(data_list, test_idx_list)
    return train_list, val_list, test_list


def get_dataset(config, data_list, split="train"):
    return RNADesignDataset(
        data_list=data_list,
        split=split,
        radius=config.radius,
        top_k=config.top_k,
        num_rbf=config.num_rbf,
        num_posenc=config.num_posenc,
        max_num_conformers=config.max_num_conformers,
        noise_scale=config.noise_scale
    )


def get_model(config):
    return GVPDiff(config)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def reward_fn(rhofold, dataset, raw_data, pred_seq, device, parallel_id=None):
    """Compute downstream reward using structural evaluation metrics."""

    results = evaluate(
        rhofold,
        dataset,
        raw_data,
        pred_seq,
        1,
        device=device,
        parallel_id=parallel_id,
    )
    score_gddt = results['sc_score_gddt'][0]
    score_tm = results['sc_score_tm'][0]
    score_rmsd = results['sc_score_rmsd'][0]

    reward = -(score_rmsd * 0.5) ** 2 + (score_gddt * 5) ** 2

    if score_gddt > 0.45:
        reward += (score_gddt - 0.45) * 100
    elif score_rmsd < 3:
        reward += (3 - score_rmsd) * 20

    return reward, score_gddt, score_tm, score_rmsd


def sample_once(
    raw_data,
    dataset,
    model,
    rhofold,
    noise_scheduler,
    config,
    autocast,
    device,
    parallel_id=None,
    deterministic=False,
    temperature=None,
):
    """Generate a single trajectory sample and return it with reward statistics."""

    data = dataset.featurizer(raw_data).to(device)
    sample_temperature = config.temperature if temperature is None else temperature
    with autocast():
        x0_pred, log_probs_traj, latents_traj = ddim_sample_with_logprob(
            model,
            noise_scheduler,
            data,
            n_steps=config.n_steps,
            device=device,
            temperature=sample_temperature,
            deterministic=deterministic,
        )
    pred_seq = torch.argmax(x0_pred, dim=-1)

    reward, score_gddt, score_tm, score_rmsd = reward_fn(
        rhofold,
        dataset,
        raw_data,
        pred_seq,
        device=device,
        parallel_id=parallel_id,
    )

    trajectory = TrajectorySample(
        log_probs_traj=log_probs_traj,
        latents_traj=latents_traj,
        reward=reward,
        raw_data=raw_data,
    )
    return trajectory, score_gddt, score_tm, score_rmsd


def collect_parallel_samples(raw_data, dataset, model, rhofold, noise_scheduler, config, autocast, device):
    """Launch additional sampling tasks in parallel to diversify exploration."""

    n_parallel_rollouts = max(config.rollouts_per_round - 1, 0)
    if n_parallel_rollouts == 0:
        return []

    samples = []
    with ThreadPoolExecutor(max_workers=n_parallel_rollouts) as executor:
        futures = [
            executor.submit(
                sample_once,
                raw_data,
                dataset,
                model,
                rhofold,
                noise_scheduler,
                config,
                autocast,
                device,
                parallel_id=parallel_id,
                deterministic=config.deterministic,
            )
            for parallel_id in range(n_parallel_rollouts)
        ]
        for future in as_completed(futures):
            sample, _, _, _ = future.result()
            samples.append(sample)
    return samples


def train_diffusion_rl(config, model, dataset, device, accelerator, optimizer):
    """Fine-tune the diffusion model with trajectory-level RL updates."""

    autocast = accelerator.autocast
    clip_range = config.clip_range

    device_rho = torch.device("cuda")
    rhofold = RhoFold(rhofold_config, device_rho)
    rhofold_path = os.path.join(PROJECT_PATH, "tools/rhofold/model_20221010_params.pt")
    print(f"Loading RhoFold checkpoint: {rhofold_path}")
    rhofold.load_state_dict(torch.load(rhofold_path, map_location=torch.device("cpu"))["model"])
    rhofold = rhofold.to(device_rho)
    rhofold.eval()

    noise_scheduler = NoiseScheduleVP(
        config.sde_schedule,
        continuous_beta_0=config.continuous_beta_0,
        continuous_beta_1=config.continuous_beta_1,
        dtype=torch.float32,
    )
    noise_scheduler.eps = config.eps

    time_grid = torch.linspace(noise_scheduler.T, noise_scheduler.eps, config.n_steps, device=device)

    avg_reward_best = float("-inf")
    baseline = 0.0
    beta_baseline = 0.8
    ckpt_dir = _checkpoint_dir()
    if len(dataset.data_list) == 0:
        raise ValueError("RL dataset is empty after preprocessing.")
    sample_index = int(config.sample_index) % len(dataset.data_list)
    raw_data_fixed = dataset.data_list[sample_index]

    for epoch in range(config.epochs):
        model.eval()
        print(f"Sample epoch: {epoch}")

        sampled_data: List[TrajectorySample] = []
        target_samples = int(config.target_samples_per_epoch)
        with tqdm(total=target_samples, desc=f"Epoch {epoch} Sampling", leave=False) as pbar:
            while len(sampled_data) < target_samples:
                # Anchor each round with one deterministic rollout, then expand with stochastic ones.
                deterministic_sample, score_gddt, score_tm, score_rmsd = sample_once(
                    raw_data_fixed,
                    dataset,
                    model,
                    rhofold,
                    noise_scheduler,
                    config,
                    autocast,
                    device,
                    deterministic=True,
                    temperature=0.0,
                )
                sampled_data.append(deterministic_sample)
                wandb.log(
                    {
                        "Test/reward": deterministic_sample.reward,
                        "Test/score_gddt": score_gddt,
                        "Test/score_tm": score_tm,
                        "Test/score_rmsd": score_rmsd,
                    }
                )

                extra_samples = collect_parallel_samples(
                    raw_data_fixed, dataset, model, rhofold, noise_scheduler, config, autocast, device
                )
                sampled_data.extend(extra_samples)

                current_count = min(len(sampled_data), target_samples)
                pbar.n = current_count
                if sampled_data:
                    recent_rewards = [s.reward for s in sampled_data[:current_count]]
                    pbar.set_postfix(avg_reward=f"{np.mean(recent_rewards):.3f}")
                pbar.refresh()

        sampled_data = sampled_data[:target_samples]
        rewards = [sample.reward for sample in sampled_data]
        mean_reward = np.mean(rewards)
        baseline = _update_baseline(baseline, mean_reward, epoch, beta_baseline)

        for sample in sampled_data:
            sample.set_advantage(baseline)

        random.shuffle(sampled_data)

        rl_losses: List[float] = []
        for _ in range(config.rl_update_epochs):
            for sample in sampled_data:
                traj_losses = []
                old_log_probs_traj = sample.log_probs_traj
                latents_traj = sample.latents_traj[:-1]
                next_latents_traj = sample.latents_traj[1:]

                total_steps = len(old_log_probs_traj)
                num_chunks = config.num_chunks
                chunk_size = (total_steps + num_chunks - 1) // num_chunks
                data = dataset.featurizer(sample.raw_data).to(device)

                for chunk_id in range(num_chunks):
                    start = chunk_id * chunk_size
                    end = min(start + chunk_size, total_steps)
                    if start >= end:
                        break

                    with accelerator.accumulate(model):
                        with autocast():
                            old_lp_chunk = torch.stack(old_log_probs_traj[start:end], dim=0).to(device)
                            z_t_chunk = torch.stack(latents_traj[start:end], dim=0).to(device)
                            z_tp1_chunk = torch.stack(next_latents_traj[start:end], dim=0).to(device)

                            t_chunk = time_grid[start:end]
                            t_next_chunk = time_grid[start + 1 : end + 1]
                            alpha_t, sigma_t = noise_scheduler.marginal_prob(t_chunk.unsqueeze(-1))
                            alpha_tp1, sigma_tp1 = noise_scheduler.marginal_prob(t_next_chunk.unsqueeze(-1))
                            noise_levels = torch.log(alpha_t**2 / sigma_t**2).to(device)

                            data.z_t = z_t_chunk
                            pred_noise = model.sample(
                                data,
                                n_samples=(end - start),
                                time=None,
                                noise_level=noise_levels,
                            )

                            x0_hat = (z_t_chunk - sigma_t.view(-1, 1, 1) * pred_noise) / alpha_t.view(-1, 1, 1)
                            mu = alpha_tp1.view(-1, 1, 1) * x0_hat
                            std = sigma_tp1.view(-1, 1, 1)

                            logp = -((z_tp1_chunk - mu) ** 2) / (2 * std**2) - std.log() - 0.5 * math.log(2 * math.pi)
                            new_lp = logp.view(end - start, -1).mean(dim=1)

                            ratio = torch.exp(new_lp - old_lp_chunk)

                            # clipped objective on trajectory chunks.
                            unclipped = -sample.advantage * ratio
                            clipped = -sample.advantage * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                            chunk_loss = torch.max(unclipped, clipped).mean()
                            if not chunk_loss.requires_grad:
                                raise RuntimeError(
                                    "chunk_loss has no grad_fn. "
                                    "Check that model forward path is not under torch.no_grad()."
                                )

                            accelerator.backward(chunk_loss)
                            if accelerator.sync_gradients:
                                utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                                optimizer.step()
                                optimizer.zero_grad()

                            traj_losses.append(chunk_loss.item())

                if traj_losses:
                    rl_losses.append(float(np.mean(traj_losses)))

        avg_loss = float(np.mean(rl_losses)) if rl_losses else 0.0
        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        logger.info(f"Epoch {epoch}: RL Loss = {avg_loss:.4f}, Avg Reward = {avg_reward:.4f}")
        wandb.log(
            {
                "train/rl_loss": avg_loss,
                "train/avg_reward": avg_reward,
                "train/lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            }
        )

        if config.save and accelerator.is_main_process and avg_reward >= avg_reward_best:
            avg_reward_best = avg_reward
            checkpoint_path = os.path.join(ckpt_dir, "current_checkpoint_rl.h5")
            torch.save(model.state_dict(), checkpoint_path)
            wandb.run.summary["best_checkpoint_rl"] = checkpoint_path

    logger.info("RL fine-tuning finished.")


def main(config, device):
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.accumulate_steps * config.num_chunks
    )

    set_seed(config.seed)

    model = get_model(config).to(device)
    total_param = sum(np.prod(list(p.size())) for p in model.parameters())
    wandb.run.summary["total_param"] = total_param

    if config.model_path:
        _load_model_checkpoint(model, config.model_path, device)

    params_to_update = [param for param in model.parameters() if param.requires_grad]

    optimizer = Adam(params_to_update, lr=config.lr)

    _, _, test_list = get_data_splits(split_type=config.split)
    dataset = get_dataset(config, test_list, split="train")

    model, optimizer = accelerator.prepare(model, optimizer)

    train_diffusion_rl(config, model, dataset, device, accelerator, optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='configs/default_rl.yaml', type=str)
    parser.add_argument('--expt_name', dest='expt_name', default=None, type=str)
    parser.add_argument('--tags', nargs='+', dest='tags', default=[])
    parser.add_argument('--no_wandb', action="store_true")
    args, _ = parser.parse_known_args()

    wandb.init(
        project=os.environ.get("WANDB_PROJECT"),
        entity=os.environ.get("WANDB_ENTITY"),
        config=args.config,
        name=args.expt_name,
        tags=args.tags,
        mode="disabled" if args.no_wandb else "online",
    )
    config = wandb.config

    config_dict = dict(wandb.config)
    config = ml_collections.ConfigDict(config_dict)

    device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")

    main(config, device)
