"""
First-order (FOGD) and zeroth-order (ZOGD) training loops in function space.
"""

from typing import List, Optional
import torch


def train_fo_loop_with_f0(
    f_0: torch.Tensor,
    y: torch.Tensor,
    ntk: torch.Tensor,
    eta: float,
    sample_num: int,
    iteration_total: int,
    record_iterations: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
) -> tuple:
    """
    FOGD: f_{t+1} = f_t - (eta/n) * K_fo^T @ (f_t - y).
    f_0: (sample_num, 1) or (sample_num,).
    Returns loss_list, f_list (at record_iterations), f_diff_list, f_final.
    """
    if device is None:
        device = f_0.device
    f_0 = f_0.to(device).reshape(-1, 1)
    y = y.to(device).reshape(-1, 1)
    ntk = ntk.to(device)
    record_iterations = set(record_iterations or [])
    loss_list = []
    f_list = []
    f_diff_list = []
    f = f_0.clone()
    for i in range(iteration_total):
        loss = (0.5 * (f - y).pow(2)).mean()
        loss_list.append(loss.item())
        f_new = f - (eta / sample_num) * (ntk.T @ (f - y))
        f_diff = f_new - f
        f_diff_list.append(f_diff.detach())
        if i in record_iterations:
            f_list.append(f.detach().clone())
        f = f_new
    loss_list = torch.tensor(loss_list)
    f_list = torch.cat(f_list, dim=0) if f_list else None
    f_diff_list = torch.cat(f_diff_list, dim=0)
    return loss_list, f_list, f_diff_list, f


def train_zo_loop_with_f0(
    f_0: torch.Tensor,
    y: torch.Tensor,
    nzk: torch.Tensor,
    eta: float,
    sample_num: int,
    iteration_total: int,
    record_iterations: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
) -> tuple:
    """
    ZOGD: f_{t+1} = f_t - (eta/n) * K_zo^T @ (f_t - y).
    Returns loss_list, f_list, f_diff_list, f_final.
    """
    if device is None:
        device = f_0.device
    f_0 = f_0.to(device).reshape(-1, 1)
    y = y.to(device).reshape(-1, 1)
    nzk = nzk.to(device)
    record_iterations = set(record_iterations or [])
    loss_list = []
    f_list = []
    f_diff_list = []
    f = f_0.clone()
    for i in range(iteration_total):
        loss = (0.5 * (f - y).pow(2)).mean()
        loss_list.append(loss.item())
        f_new = f - (eta / sample_num) * (nzk.T @ (f - y))
        f_diff = f_new - f
        f_diff_list.append(f_diff.detach())
        if i in record_iterations:
            f_list.append(f.detach().clone())
        f = f_new
    loss_list = torch.tensor(loss_list)
    f_list = torch.cat(f_list, dim=0) if f_list else None
    f_diff_list = torch.cat(f_diff_list, dim=0)
    return loss_list, f_list, f_diff_list, f


def _l2_loss_linear(x: torch.Tensor, theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """L2 loss in linearized space: 0.5 * mean((x @ theta - y)^2)."""
    return (0.5 * (x @ theta - y).pow(2)).mean()


def train_zo_standard_loop(
    x: torch.Tensor,
    y: torch.Tensor,
    theta_0: torch.Tensor,
    eta: float,
    iteration_total: int,
    epsilon: float = 1e-3,
    n_z_samples: int = 100,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Standard zeroth-order in parameter space (as in FFN notebooks): each iteration
    sample n_z_samples z, pick z_best that maximizes (L(theta-eps*z)-L(theta+eps*z)),
    update theta = theta - eta * (diff_best/(2*eps)) * z_best.
    Returns loss_list (length iteration_total).
    """
    if device is None:
        device = x.device
    x = x.to(device)
    y = y.to(device).reshape(-1, 1)
    theta = theta_0.to(device).reshape(-1, 1)
    loss_list = []
    for _ in range(iteration_total):
        loss = _l2_loss_linear(x, theta, y)
        loss_list.append(loss.item())
        diff_best = -1e10
        z_best = None
        for _ in range(n_z_samples):
            z = torch.randn(1, theta.numel(), device=device)
            z = z.reshape_as(theta)
            d_plus = _l2_loss_linear(x, theta + epsilon * z, y)
            d_minus = _l2_loss_linear(x, theta - epsilon * z, y)
            local_diff = (d_minus - d_plus).item()
            if local_diff > diff_best:
                diff_best = local_diff
                z_best = z.clone()
        if z_best is not None:
            theta = theta - eta * (diff_best / (2 * epsilon)) * z_best
    return torch.tensor(loss_list)
