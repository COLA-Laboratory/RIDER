import math

import torch


class NoiseScheduleVP:
    """VP-SDE noise schedule wrapper used by diffusion training and sampling."""

    def __init__(
        self,
        schedule="discrete",
        betas=None,
        alphas_cumprod=None,
        continuous_beta_0=0.1,
        continuous_beta_1=20.0,
        eps=1e-3,
        dtype=torch.float32,
    ):
        if schedule not in ["discrete", "linear", "cosine", "discrete_poly"]:
            raise ValueError(f"Unsupported noise schedule: {schedule}")

        self.schedule = schedule
        if "discrete" in schedule:
            if schedule == "discrete_poly":
                alphas_cumprod = get_polynomial_schedule(1000, power=2)
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            elif betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)

            self.total_N = len(log_alphas)
            self.T = 1.0
            self.t_array = torch.linspace(0.0, 1.0, self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
            self.log_alpha_array = log_alphas.reshape((1, -1)).to(dtype=dtype)
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1.0 + self.cosine_s) * math.pi / 2.0))
            self.T = 0.9946 if schedule == "cosine" else 1.0

        self.eps = eps

    def marginal_log_mean_coeff(self, t):
        if "discrete" in self.schedule:
            return interpolate_fn(
                t.reshape((-1, 1)),
                self.t_array.to(t.device),
                self.log_alpha_array.to(t.device),
            ).reshape((-1))

        if self.schedule == "linear":
            return -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

        log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1.0 + self.cosine_s) * math.pi / 2.0))
        return log_alpha_fn(t) - self.cosine_log_alpha_0

    def marginal_prob(self, t):
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        alpha_t = torch.exp(log_mean_coeff)
        sigma_t = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return alpha_t, sigma_t


def interpolate_fn(x, xp, yp):
    """Differentiable piecewise-linear interpolation."""
    n, k = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((n, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1

    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(torch.eq(x_idx, k), torch.tensor(k - 2, device=x.device), cand_start_idx),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)

    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)

    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(torch.eq(x_idx, k), torch.tensor(k - 2, device=x.device), cand_start_idx),
    )

    y_positions_expanded = yp.unsqueeze(0).expand(n, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)

    return start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)


def get_polynomial_schedule(time_steps, s=1e-4, power=2):
    """Polynomial alpha schedule used by the discrete_poly mode."""
    steps = time_steps + 1
    x = torch.linspace(0, steps, steps)
    alphas2 = (1 - torch.pow(x / steps, power)) ** 2

    alphas2 = torch.cat([torch.ones(1), alphas2], dim=0)
    alphas_step = alphas2[1:] / alphas2[:-1]
    alphas_step = torch.clamp(alphas_step, min=0.001, max=1.0)
    alphas2 = torch.cumprod(alphas_step, dim=0)

    precision = 1 - 2 * s
    alphas2 = precision * alphas2 + s
    return alphas2[1:]
