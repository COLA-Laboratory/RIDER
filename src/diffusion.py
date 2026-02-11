import math

import torch
from tqdm import tqdm

from src.constants import NUM_TO_LETTER
from src.noise_schedule import NoiseScheduleVP


def build_noise_scheduler(config) -> NoiseScheduleVP:
    """Create a diffusion noise scheduler for inference/sampling."""
    return NoiseScheduleVP(
        config.sde_schedule,
        continuous_beta_0=config.continuous_beta_0,
        continuous_beta_1=config.continuous_beta_1,
        eps=config.eps,
    )


def evaluate_sampling_per_sample(model, dataset, noise_scheduler, n_steps, device, temperature=1.0):
    """Sample each RNA and report token-level sequence accuracy."""
    model.eval()
    total_correct = 0
    total_count = 0
    samples_list = []

    with torch.no_grad():
        for idx, raw_data in enumerate(dataset.data_list):
            seq_length = len(raw_data["sequence"])
            data = dataset.featurizer(raw_data).to(device)

            x0_pred, _, _ = ddim_sample_with_logprob(
                model,
                noise_scheduler,
                data,
                n_steps,
                device,
                temperature,
            )

            pred_seq = torch.argmax(x0_pred, dim=-1)
            true_seq = torch.tensor([int(c) for c in data["seq"]], device=device)

            correct = (pred_seq == true_seq).sum().item()
            total_correct += correct
            total_count += seq_length

            samples_list.append(pred_seq.cpu().numpy())
            print(f"Sample {idx} Accuracy: {correct / seq_length:.4f}")

    overall_acc = total_correct / total_count if total_count > 0 else 0.0
    print(f"\nOverall Sampling Accuracy: {overall_acc:.4f}")
    return overall_acc, samples_list


def _seq_tensor_to_letters(seq_tensor: torch.Tensor) -> str:
    indices = seq_tensor.detach().cpu().tolist()
    return "".join(NUM_TO_LETTER[int(idx)] for idx in indices)


def evaluate_single_sample_best_of_n(
    model,
    dataset,
    raw_data,
    noise_scheduler,
    n_steps,
    device,
    n_repeats=5,
    temperature=0.5,
    deterministic=False,
):
    """Run repeated sampling on one sample and return the best prediction by accuracy."""
    model.eval()
    data = dataset.featurizer(raw_data).to(device)
    true_seq = torch.tensor([int(c) for c in data["seq"]], device=device)

    best_acc = -1.0
    best_pred_seq = None

    with torch.no_grad():
        for _ in range(max(int(n_repeats), 1)):
            x0_pred, _, _ = ddim_sample_with_logprob(
                model,
                noise_scheduler,
                data,
                n_steps,
                device,
                temperature=temperature,
                deterministic=deterministic,
            )
            pred_seq = torch.argmax(x0_pred, dim=-1)
            acc = (pred_seq == true_seq).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_pred_seq = pred_seq.clone()

    designed_seq = _seq_tensor_to_letters(best_pred_seq)
    return {
        "best_acc": best_acc,
        "designed_seq": designed_seq,
    }


def evaluate_dataset_best_of_n(
    model,
    dataset,
    noise_scheduler,
    n_steps,
    device,
    n_repeats=5,
    temperature=0.5,
    deterministic=False,
):
    """Evaluate all samples and average per-sample best-of-n accuracy."""
    best_accs = []
    iterator = tqdm(enumerate(dataset.data_list), total=len(dataset.data_list), desc="Evaluating test set")
    for _, raw_data in iterator:
        result = evaluate_single_sample_best_of_n(
            model=model,
            dataset=dataset,
            raw_data=raw_data,
            noise_scheduler=noise_scheduler,
            n_steps=n_steps,
            device=device,
            n_repeats=n_repeats,
            temperature=temperature,
            deterministic=deterministic,
        )
        best_accs.append(result["best_acc"])
        iterator.set_postfix(avg_best_acc=f"{sum(best_accs) / len(best_accs):.4f}")

    avg_best_acc = sum(best_accs) / len(best_accs) if best_accs else 0.0
    return {"avg_best_acc": avg_best_acc}


def ddim_sample_with_logprob(
    model,
    noise_scheduler,
    data,
    n_steps,
    device,
    temperature=0.5,
    deterministic=False,
):
    """DDIM sampling that also returns transition log-probabilities."""
    model.eval()
    log_probs = []
    latents = []

    with torch.no_grad():
        num_nodes = data.seq.shape[0]
        out_dim = model.out_dim
        x = torch.randn(num_nodes, out_dim, device=device)

        T = noise_scheduler.T
        eps = noise_scheduler.eps
        time_grid = torch.linspace(T, eps, n_steps, device=device)

        for i in range(len(time_grid) - 1):
            t = time_grid[i]
            t_next = time_grid[i + 1]
            t_tensor = t.unsqueeze(0)
            t_next_tensor = t_next.unsqueeze(0)

            alpha_t, sigma_t = noise_scheduler.marginal_prob(t_tensor)
            noise_level = torch.log(alpha_t**2 / sigma_t**2)

            data.z_t = x
            pred_noise = model(
                data,
                time=t_tensor.unsqueeze(0),
                noise_level=noise_level.unsqueeze(0),
            ).squeeze(0)

            x0_hat = (x - sigma_t * pred_noise) / alpha_t
            alpha_next, sigma_next = noise_scheduler.marginal_prob(t_next_tensor)

            if deterministic:
                noise = torch.zeros_like(x)
            else:
                noise = torch.randn_like(x) * temperature

            x_next = alpha_next * x0_hat + sigma_next * noise

            mu = alpha_next * x0_hat
            std = sigma_next
            var = std**2
            log_prob = -((x_next - mu) ** 2) / (2 * var) - std.log() - 0.5 * math.log(2 * math.pi)
            log_prob = log_prob.mean()

            latents.append(x)
            log_probs.append(log_prob)
            x = x_next

        latents.append(x)
        return x, log_probs, latents
