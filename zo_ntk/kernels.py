"""
NTK (Neural Tangent Kernel) and NZK (Neural Zeroth-order Kernel) construction.
"""

from typing import Optional, Literal
import torch


def build_ntk(x: torch.Tensor) -> torch.Tensor:
    """
    First-order NTK: K_fo = X X^T (Gram matrix of feature vectors).
    Works for linear features (data_sample) or Jacobian features (x_preprocessed).
    """
    return x @ x.T


def build_nzk_linear(
    data_sample: torch.Tensor,
    n_samples: int = 10000,
    dist: Literal["normal", "t", "laplace"] = "normal",
    scale: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    NZK for linear case: E_z[ (z^T x_i)(z^T x_j)(z^T z) ].
    data_sample: (sample_num, degree).
    """
    if device is None:
        device = data_sample.device
    sample_num, degree = data_sample.shape
    x = data_sample.to(device)
    nzk_list = []
    for _ in range(n_samples):
        if dist == "normal":
            z = torch.normal(0.0, scale, size=(1, degree), device=device)
        elif dist == "t":
            import numpy as np
            df = max(1, int(scale)) if scale >= 1 else 10
            z = torch.tensor(
                np.random.standard_t(df, size=(1, degree)),
                dtype=torch.float32,
                device=device,
            )
        elif dist == "laplace":
            import numpy as np
            z = torch.tensor(
                np.random.laplace(0, scale, size=(1, degree)),
                dtype=torch.float32,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported distribution: {dist}")
        z = z.repeat(sample_num, 1)
        first = z @ x.T
        second = z @ x.T
        third = z @ z.T
        nzk_temp = first.T * second * third
        nzk_list.append(nzk_temp.unsqueeze(0))
    nzk = torch.cat(nzk_list, dim=0).mean(dim=0)
    return nzk.cpu() if nzk.device != data_sample.device else nzk


def build_nzk(
    x: torch.Tensor,
    n_samples: int = 10000,
    scale: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    NZK in feature space (e.g. Jacobian features): same formula as linear,
    with x of shape (sample_num, feature_dim).
    """
    if device is None:
        device = x.device
    sample_num, feature_dim = x.shape
    x_dev = x.to(device)
    nzk_list = []
    for _ in range(n_samples):
        z = torch.normal(0.0, scale, size=(1, feature_dim), device=device)
        z = z.repeat(sample_num, 1)
        first = z @ x_dev.T
        second = z @ x_dev.T
        third = z @ z.T
        nzk_temp = first.T * second * third
        nzk_list.append(nzk_temp.unsqueeze(0))
    nzk = torch.cat(nzk_list, dim=0).mean(dim=0)
    return nzk.cpu() if nzk.device != x.device else nzk
