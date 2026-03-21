"""
Zero-Order Gradient Descent with NTK/NZK kernels.
Academic open-source implementation for zeroth-order optimization in kernel space.
"""

from .kernels import build_ntk, build_nzk_linear, build_nzk
from .training import train_fo_loop_with_f0, train_zo_loop_with_f0, train_zo_standard_loop
from .models import LinearTarget, FFN, target_linear_function
from .data import (
    synthetic_linear_data,
    linearize_ffn,
    load_mnist_binary,
    load_cifar_binary,
    load_imagenet_binary,
)

__all__ = [
    "build_ntk",
    "build_nzk_linear",
    "build_nzk",
    "train_fo_loop_with_f0",
    "train_zo_loop_with_f0",
    "train_zo_standard_loop",
    "LinearTarget",
    "FFN",
    "target_linear_function",
    "synthetic_linear_data",
    "linearize_ffn",
    "load_mnist_binary",
    "load_cifar_binary",
    "load_imagenet_binary",
]
