"""
Data loading and preprocessing: synthetic linear, MNIST/CIFAR/ImageNet binary, FFN linearization.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


def synthetic_linear_data(
    sample_num: int,
    degree: int,
    noise_std: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample x on unit sphere, theta_gt random, target_gt = x @ theta_gt + noise.
    Returns data_sample, theta_gt, target_gt, noise_value.
    """
    if seed is not None:
        torch.manual_seed(seed)
    data_sample = torch.rand(size=(sample_num, degree))
    data_sample = data_sample / data_sample.norm(dim=1, keepdim=True, p=2)
    noise_value = torch.randn(sample_num, 1) * noise_std
    theta_gt = torch.randint(1, 10, (degree, 1), dtype=torch.float32)
    target_gt = data_sample @ theta_gt + noise_value
    return data_sample, theta_gt, target_gt, noise_value


def linearize_ffn(
    network: nn.Module,
    train_data: torch.Tensor,
    normalize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Jacobian features at current parameters: each row = grad_theta f(x_i).
    Returns x_preprocessed (sample_num, param_dim), theta_0 (flattened params).
    """
    theta_0 = torch.cat([p.reshape(-1) for p in network.parameters()])
    x_preprocessed = []
    for x_ in train_data:
        network.zero_grad()
        x_pred = network(x_.unsqueeze(0))
        if x_pred.dim() > 1:
            x_pred = x_pred.squeeze(0)
        grad_list = torch.autograd.grad(
            x_pred,
            network.parameters(),
            grad_outputs=torch.ones_like(x_pred),
            create_graph=False,
        )
        x_preprocessed.append(torch.cat([g.reshape(-1) for g in grad_list]))
    x_preprocessed = torch.stack(x_preprocessed)
    if normalize:
        x_preprocessed = x_preprocessed / (x_preprocessed.norm(dim=1, keepdim=True, p=2) + 1e-8)
    return x_preprocessed, theta_0


def load_mnist_binary(
    digit_neg: int = 3,
    digit_pos: int = 5,
    sample_per_class: int = 100,
    resolution: int = 8,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load MNIST binary (e.g. 3 vs 5), resize to resolution x resolution, flatten."""
    from datasets import load_dataset
    import numpy as np
    import cv2

    if seed is not None:
        np.random.seed(seed)
    ds = load_dataset("mnist", split="train").with_format("np")
    img_key = "image"
    idx_neg = np.random.choice(
        np.where(np.array(ds["label"]) == digit_neg)[0], sample_per_class
    )
    idx_pos = np.random.choice(
        np.where(np.array(ds["label"]) == digit_pos)[0], sample_per_class
    )
    imgs_neg = np.array([ds[int(i)][img_key] for i in idx_neg])
    imgs_pos = np.array([ds[int(i)][img_key] for i in idx_pos])
    if imgs_neg[0].ndim == 2:
        imgs_neg = imgs_neg / 255.0
        imgs_pos = imgs_pos / 255.0
    else:
        imgs_neg = imgs_neg / 255.0
        imgs_pos = imgs_pos / 255.0
    imgs_neg = np.array([cv2.resize(im, (resolution, resolution)) for im in imgs_neg]).reshape(-1, resolution * resolution)
    imgs_pos = np.array([cv2.resize(im, (resolution, resolution)) for im in imgs_pos]).reshape(-1, resolution * resolution)
    X = np.concatenate([imgs_neg, imgs_pos], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(sample_per_class), np.ones(sample_per_class)]).astype(np.float32)
    perm = np.random.permutation(X.shape[0])
    X = torch.from_numpy(X[perm])
    y = torch.from_numpy(y[perm]).reshape(-1, 1)
    return X, y


def load_cifar_binary(
    digit_neg: int = 2,
    digit_pos: int = 9,
    sample_per_class: int = 100,
    resolution: int = 8,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load CIFAR-10 binary, grayscale, resize, flatten."""
    from datasets import load_dataset
    import numpy as np
    import cv2

    if seed is not None:
        np.random.seed(seed)
    ds = load_dataset("uoft-cs/cifar10", split="train").with_format("np")
    img_key = "img"
    labels = np.array(ds["label"])
    idx_neg = np.random.choice(np.where(labels == digit_neg)[0], sample_per_class)
    idx_pos = np.random.choice(np.where(labels == digit_pos)[0], sample_per_class)
    imgs_neg = np.array([ds[int(i)][img_key] for i in idx_neg])
    imgs_pos = np.array([ds[int(i)][img_key] for i in idx_pos])
    if imgs_neg[0].ndim == 3:
        imgs_neg = np.array([cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in imgs_neg]) / 255.0
        imgs_pos = np.array([cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in imgs_pos]) / 255.0
    imgs_neg = np.array([cv2.resize(im, (resolution, resolution)) for im in imgs_neg]).reshape(-1, resolution * resolution)
    imgs_pos = np.array([cv2.resize(im, (resolution, resolution)) for im in imgs_pos]).reshape(-1, resolution * resolution)
    X = np.concatenate([imgs_neg, imgs_pos], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(sample_per_class), np.ones(sample_per_class)]).astype(np.float32)
    perm = np.random.permutation(X.shape[0])
    X = torch.from_numpy(X[perm])
    y = torch.from_numpy(y[perm]).reshape(-1, 1)
    return X, y


def load_imagenet_binary(
    class_neg: int = 0,
    class_pos: int = 1,
    sample_per_class: int = 100,
    resolution: int = 8,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load Tiny ImageNet binary, grayscale, resize, flatten."""
    from datasets import load_dataset
    import numpy as np
    import cv2

    if seed is not None:
        np.random.seed(seed)
    ds = load_dataset("zh-plus/tiny-imagenet", split="train").with_format("np")
    img_key = "image"
    labels = np.array(ds["label"])
    idx_neg = np.random.choice(np.where(labels == class_neg)[0], sample_per_class)
    idx_pos = np.random.choice(np.where(labels == class_pos)[0], sample_per_class)
    imgs_neg = np.array([ds[int(i)][img_key] for i in idx_neg])
    imgs_pos = np.array([ds[int(i)][img_key] for i in idx_pos])
    if imgs_neg[0].ndim == 3:
        imgs_neg = np.array([cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in imgs_neg]) / 255.0
        imgs_pos = np.array([cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in imgs_pos]) / 255.0
    imgs_neg = np.array([cv2.resize(im, (resolution, resolution)) for im in imgs_neg]).reshape(-1, resolution * resolution)
    imgs_pos = np.array([cv2.resize(im, (resolution, resolution)) for im in imgs_pos]).reshape(-1, resolution * resolution)
    X = np.concatenate([imgs_neg, imgs_pos], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(sample_per_class), np.ones(sample_per_class)]).astype(np.float32)
    perm = np.random.permutation(X.shape[0])
    X = torch.from_numpy(X[perm])
    y = torch.from_numpy(y[perm]).reshape(-1, 1)
    return X, y
