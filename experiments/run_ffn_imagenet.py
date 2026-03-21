#!/usr/bin/env python3
"""
FFN ZOGD/NTK on Tiny ImageNet binary classification.
Reproduces FFN_zo_fo_imagenet.
"""

import argparse
import os
import copy
import torch
from zo_ntk import (
    load_imagenet_binary,
    FFN,
    linearize_ffn,
    build_ntk,
    build_nzk,
    train_fo_loop_with_f0,
    train_zo_loop_with_f0,
    train_zo_standard_loop,
)
from zo_ntk.utils import save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_neg", type=int, default=0)
    parser.add_argument("--class_pos", type=int, default=1)
    parser.add_argument("--sample_per_class", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--iteration_total", type=int, default=16000)
    parser.add_argument("--eta_fo", type=float, default=1e-3)
    parser.add_argument("--eta_zo", type=float, default=1e-3)
    parser.add_argument("--nzk_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="output_ffn_imagenet")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_data, train_label = load_imagenet_binary(
        args.class_neg, args.class_pos, args.sample_per_class, args.resolution, args.seed
    )
    train_data = train_data.to(device)
    train_label = train_label.to(device)
    input_dim = args.resolution * args.resolution
    network = FFN(input_dim=input_dim, hidden_dim=[10, 5], output_dim=1).to(device)

    x_preprocessed, theta_0 = linearize_ffn(network, train_data, normalize=False)
    x_preprocessed = x_preprocessed.to(device)
    y = train_label
    sample_num = x_preprocessed.shape[0]
    record_list = [0, 99, 999, 3999, 4999, 5999, 8999, 9999, args.iteration_total - 1]

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

    standard_loss_list_zo = train_zo_standard_loop(
        x_preprocessed, y, theta_0, args.eta_zo, args.iteration_total,
        epsilon=1e-3, n_z_samples=100, device=device
    )

    output_folder = os.path.join(
        args.out_dir, f"imagenet_{args.resolution}_{args.class_neg}_{args.class_pos}"
    )
    os.makedirs(output_folder, exist_ok=True)
    save_results(output_folder, {
        "f_fo_list": f_fo_list,
        "f_zo_list": f_zo_list,
        "loss_list_fo": loss_fo,
        "loss_list_zo": loss_zo,
        "standard_loss_list_zo": standard_loss_list_zo,
        "f_diff_list_fo": f_diff_fo,
        "f_diff_list_zo": f_diff_zo,
        "function_record_iteration_list": record_list,
        "NTK_kernal_fo": NTK_fo.cpu(),
        "NZK_kernal_zo": NZK_zo.cpu(),
    })
    print(f"Saved to {output_folder}")


if __name__ == "__main__":
    main()
