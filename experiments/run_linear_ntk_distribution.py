#!/usr/bin/env python3
"""
Linear ZOGD with different z distributions for NZK: Normal, Student-t, Laplace.
Reproduces linear_zo_fo_ntk_distribution.
"""

import argparse
import os
import torch
from zo_ntk import (
    synthetic_linear_data,
    build_ntk,
    build_nzk_linear,
    train_fo_loop_with_f0,
    train_zo_loop_with_f0,
)
from zo_ntk.utils import save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--iteration_total", type=int, default=16000)
    parser.add_argument("--eta_fo", type=float, default=1e-3)
    parser.add_argument("--eta_zo", type=float, default=1e-3)
    parser.add_argument("--nzk_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="output_linear_ntk_distribution")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # (distribution_name, scale) for NZK
    configs = [
        ("normal", 1.0),
        ("t", 10.0),
        ("laplace", 0.5),
        ("t", 1000.0),
        ("laplace", 0.605),
    ]
    output_folder = os.path.join(args.out_dir, f"distribution_mixture_degree_{args.degree}")
    os.makedirs(output_folder, exist_ok=True)

    data_sample, theta_gt, target_gt, noise_value = synthetic_linear_data(
        args.sample_num, args.degree, noise_std=0.02, seed=args.seed
    )
    theta_init = torch.randn(args.degree, 1)
    sample_num = data_sample.shape[0]
    record_list = [0, 49, 99, 199, 999, 1999, 2999, 3999, args.iteration_total - 1]

    NTK_fo = build_ntk(data_sample)
    f_0_fo = data_sample @ theta_init
    loss_fo, f_fo_list, f_diff_fo, _ = train_fo_loop_with_f0(
        f_0_fo, target_gt, NTK_fo, args.eta_fo, sample_num,
        args.iteration_total, record_list, device
    )

    NZK_dict = {}
    for dist, scale in configs:
        nzk = build_nzk_linear(
            data_sample, n_samples=args.nzk_samples,
            dist=dist, scale=scale, device=device
        )
        NZK_dict[f"{dist}_{scale}"] = nzk

    f_zo_dict = {}
    loss_zo_dict = {}
    f_diff_zo_dict = {}
    for key, NZK_zo in NZK_dict.items():
        f_0_zo = data_sample @ theta_init
        loss_zo, f_zo_list, f_diff_zo, _ = train_zo_loop_with_f0(
            f_0_zo, target_gt, NZK_zo, args.eta_zo, sample_num,
            args.iteration_total, record_list, device
        )
        f_zo_dict[key] = f_zo_list
        loss_zo_dict[key] = loss_zo
        f_diff_zo_dict[key] = f_diff_zo

    out = {
        "theta_gt": theta_gt,
        "data_sample": data_sample,
        "target_gt": target_gt,
        "noise_value": noise_value,
        "f_fo_list": f_fo_list,
        "loss_list_fo": loss_fo,
        "f_diff_list_fo": f_diff_fo,
        "function_record_iteration_list": record_list,
        "NTK_kernal_fo": NTK_fo,
    }
    for k, v in f_zo_dict.items():
        out[f"f_zo_list_{k}"] = v
    for k, v in loss_zo_dict.items():
        out[f"loss_list_zo_{k}"] = v
    for k, v in f_diff_zo_dict.items():
        out[f"f_diff_list_zo_{k}"] = v
    for k, v in NZK_dict.items():
        out[f"NZK_kernal_zo_{k}"] = v

    save_results(output_folder, out)
    print(f"Saved to {output_folder}")


if __name__ == "__main__":
    main()
