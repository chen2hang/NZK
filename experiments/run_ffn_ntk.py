#!/usr/bin/env python3
"""
FFN ZOGD/NTK on synthetic data (sphere + FFN linearization).
Reproduces FFN_zo_fo_ntk.
"""

import argparse
import os
import copy
import torch
import torch.nn as nn
from zo_ntk import (
    build_ntk,
    build_nzk,
    train_fo_loop_with_f0,
    train_zo_loop_with_f0,
    FFN,
    linearize_ffn,
)
from zo_ntk import synthetic_linear_data
from zo_ntk.utils import save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--sample_num", type=int, default=150)
    parser.add_argument("--hidden_dim", type=int, default=5)
    parser.add_argument("--iteration_total", type=int, default=16000)
    parser.add_argument("--eta_fo", type=float, default=1e-3)
    parser.add_argument("--eta_zo", type=float, default=1e-3)
    parser.add_argument("--nzk_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="output_ffn_ntk")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    data_sample, theta_gt, target_gt, _ = synthetic_linear_data(
        args.sample_num, args.degree, noise_std=0.05, seed=args.seed
    )
    data_sample = data_sample.to(device)
    target_gt = target_gt.to(device)

    network = FFN(input_dim=args.degree, hidden_dim=args.hidden_dim, output_dim=1).to(device)
    x_preprocessed, theta_0 = linearize_ffn(network, data_sample, normalize=True)
    x_preprocessed = x_preprocessed.to(device)
    y = target_gt
    sample_num = x_preprocessed.shape[0]
    record_list = [0, 49, 99, 199, 999, 1999, 2999, 3999, args.iteration_total - 1]

    NTK_fo = build_ntk(x_preprocessed)
    theta_fo = copy.deepcopy(theta_0.detach()).view(-1, 1)
    f_0_fo = x_preprocessed @ theta_fo
    loss_fo, f_fo_list, f_diff_fo, _ = train_fo_loop_with_f0(
        f_0_fo, y, NTK_fo, args.eta_fo, sample_num,
        args.iteration_total, record_list, device
    )

    NZK_zo = build_nzk(x_preprocessed, n_samples=args.nzk_samples, device=device)
    theta_zo = copy.deepcopy(theta_0.detach()).view(-1, 1)
    f_0_zo = x_preprocessed @ theta_zo
    loss_zo, f_zo_list, f_diff_zo, _ = train_zo_loop_with_f0(
        f_0_zo, y, NZK_zo, args.eta_zo, sample_num,
        args.iteration_total, record_list, device
    )

    output_folder = os.path.join(args.out_dir, f"ffn_degree_{args.degree}")
    os.makedirs(output_folder, exist_ok=True)
    save_results(output_folder, {
        "data_sample": data_sample.cpu(),
        "target_gt": target_gt.cpu(),
        "f_fo_list": f_fo_list,
        "f_zo_list": f_zo_list,
        "loss_list_fo": loss_fo,
        "loss_list_zo": loss_zo,
        "f_diff_list_fo": f_diff_fo,
        "f_diff_list_zo": f_diff_zo,
        "function_record_iteration_list": record_list,
        "NTK_kernal_fo": NTK_fo.cpu(),
        "NZK_kernal_zo": NZK_zo.cpu(),
    })
    print(f"Saved to {output_folder}")


if __name__ == "__main__":
    main()
