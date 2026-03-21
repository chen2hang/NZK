"""
Models: linear target and feedforward network (FFN).
"""

from typing import List, Union
import torch
import torch.nn as nn


def target_linear_function(
    x: torch.Tensor,
    weight: torch.Tensor,
    noise_value: torch.Tensor,
    noise: bool = True,
) -> torch.Tensor:
    """Linear target: y = x @ weight + optional noise."""
    if noise:
        return x @ weight + noise_value
    return x @ weight


class LinearTarget:
    """Linear predictor f = x @ theta."""

    def __init__(self, theta: torch.Tensor):
        self.theta = theta

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.theta


class FFN(nn.Module):
    """
    Feedforward network with ReLU. Used for FFN-NTK experiments.
    hidden_dim can be int (two hidden layers of that size, matching FFN_zo_fo_ntk)
    or list of ints (e.g. [10, 5] for MNIST/CIFAR).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Union[int, List[int]] = 5,
        output_dim: int = 1,
    ):
        super().__init__()
        if isinstance(hidden_dim, int):
            # Match notebook: two hidden layers of same size (e.g. input->5->5->1)
            hidden_dim = [hidden_dim, hidden_dim]
        layers = []
        dims = [input_dim] + list(hidden_dim) + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)
