# Model Evolution Under Zeroth-Order Optimization — Official Implementation

Official implementation of **Model Evolution Under Zeroth-Order Optimization: A Neural Tangent Kernel Perspective**.

**Paper (OpenReview):** [https://openreview.net/forum?id=PCJGU7DEEX](https://openreview.net/forum?id=PCJGU7DEEX)  
**Venue:** Workshop on Scientific Methods for Understanding Deep Learning (ICLR 2026)  
**Code repository:** [https://github.com/BellChingH/NZK](https://github.com/BellChingH/NZK)

This code implements **Zeroth-Order Gradient Descent (ZOGD)** and **First-Order Gradient Descent (FOGD)** in function space using the **Neural Tangent Kernel (NTK)** and **Neural Zeroth-order Kernel (NZK)**. Each experiment script writes a single MATLAB-compatible file `data.mat` under an output subfolder; you can load it in Python (`scipy.io.loadmat`), MATLAB, or Julia to analyze or plot results.

## Structure

```
NZK/
├── zo_ntk/                 # Core library
│   ├── kernels.py          # NTK and NZK construction
│   ├── training.py         # FOGD and ZOGD loops in function space
│   ├── models.py           # Linear target, FFN
│   ├── data.py             # Synthetic data, MNIST/CIFAR/ImageNet loaders, FFN linearization
│   └── utils.py            # Saving results to data.mat
├── experiments/            # Reproducible experiment scripts (recommended)
│   ├── run_linear_ntk.py              # Linear synthetic (degree 2 or 50)
│   ├── run_linear_ntk_distribution.py # Linear, NZK with different z distributions
│   ├── run_linear_ntk_variance.py     # Linear, NZK with different Gaussian variances
│   ├── run_ffn_ntk.py                 # FFN on synthetic sphere data
│   ├── run_ffn_mnist.py               # FFN on MNIST (binary classes)
│   ├── run_ffn_cifar.py               # FFN on CIFAR-10 (binary classes)
│   └── run_ffn_imagenet.py            # FFN on Tiny ImageNet (binary classes)
├── notebooks/              # Original Jupyter experiments
├── requirements.txt
├── requirements-notebooks.txt  # Optional: Jupyter + matplotlib for notebooks/
└── README.md
```

## Installation

Clone the official repository and enter the project root (the folder that contains `zo_ntk/` and `experiments/`):

```bash
git clone https://github.com/BellChingH/NZK.git
cd NZK
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Image experiments need `datasets`, `opencv-python-headless`, and `Pillow`; they are listed in `requirements.txt`. First-time runs will download the corresponding Hugging Face datasets.

### Jupyter notebooks (optional)

The original **`.ipynb`** experiments live under **`notebooks/`**, grouped by topic (same names as the legacy folders: `linear_zo_fo_ntk`, `FFN_zo_fo_mnist`, etc.). Install extra dependencies and see usage notes:

```bash
pip install -r requirements-notebooks.txt
```

## How to run (quick start)

1. **Working directory:** Always run commands from inside the **repository root** (`NZK/`) so `python -m experiments.<script>` resolves the package correctly.

2. **List options for any script:**

   ```bash
   python -m experiments.run_linear_ntk --help
   python -m experiments.run_ffn_mnist --help
   ```

3. **Minimal examples:**

   ```bash
   cd NZK

   # Linear synthetic: NTK + NZK (one Gaussian NZK), degree 2
   python -m experiments.run_linear_ntk --degree 2 --out_dir output_linear_ntk

   # FFN on MNIST (e.g. digits 3 vs 5, 8×8 inputs)
   python -m experiments.run_ffn_mnist --digit_neg 3 --digit_pos 5 --out_dir output_ffn_mnist
   ```

4. **Output location:** Each script creates a **subfolder** under `--out_dir` (see table below) and writes **`data.mat`** there. The terminal prints `Saved to <path>` when finished.

### Commands by experiment

| Goal | Command (from repo root `NZK/`) |
|------|-------------------------------------|
| Linear synthetic, one NZK (normal) | `python -m experiments.run_linear_ntk --degree 2 --out_dir output_linear_ntk` |
| Linear synthetic, degree 50 | `python -m experiments.run_linear_ntk --degree 50 --out_dir output_linear_ntk` |
| Linear, multiple z distributions | `python -m experiments.run_linear_ntk_distribution --out_dir output_linear_ntk_distribution` |
| Linear, multiple Gaussian variances | `python -m experiments.run_linear_ntk_variance --variances 0.5 1.0 1.5 --out_dir output_linear_ntk_variance` |
| FFN synthetic | `python -m experiments.run_ffn_ntk --degree 2 --out_dir output_ffn_ntk` |
| FFN MNIST binary | `python -m experiments.run_ffn_mnist --digit_neg 3 --digit_pos 5 --out_dir output_ffn_mnist` |
| FFN CIFAR-10 binary | `python -m experiments.run_ffn_cifar --digit_neg 2 --digit_pos 9 --out_dir output_ffn_cifar` |
| FFN Tiny ImageNet binary | `python -m experiments.run_ffn_imagenet --class_neg 0 --class_pos 1 --out_dir output_ffn_imagenet` |

**Default subfolder names under `--out_dir`:** `degree_<d>` (linear / FFN synthetic), `distribution_mixture_degree_<d>` (distribution), `variance_mixture_degree_<d>` (variance), `mnist_<res>_<neg>_<pos>`, `cifar_<res>_<neg>_<pos>`, `imagenet_<res>_<neg>_<pos>`.

## Output file: `data.mat`

All arrays are saved as NumPy types inside the `.mat` file (load with `scipy.io.loadmat`). Variable names use the historical spelling **`kernal`** (not `kernel`) for NTK/NZK matrices.

### Fields common to many runs

| Variable | Meaning |
|----------|---------|
| `loss_list_fo` | Per-iteration MSE loss \((1/2n)\|f-y\|^2\) for **first-order** (NTK) dynamics; length = `iteration_total`. |
| `loss_list_zo` | Same for **zeroth-order** (NZK) dynamics (when a single ZO run is saved). |
| `f_fo_list` | Function values \(f\) along the FO trajectory, **concatenated** at iterations listed in `function_record_iteration_list` (each snapshot has shape `(sample_num, 1)`; overall first dimension is `len(function_record_iteration_list) * sample_num`). |
| `f_zo_list` | Same for the ZO trajectory (single-run scripts). |
| `f_diff_list_fo` | Per-iteration update \(f_{t+1}-f_t\) for FO; concatenated along the first axis (length `iteration_total * sample_num` in the flattened layout stored). |
| `f_diff_list_zo` | Same for ZO (single-run scripts). |
| `function_record_iteration_list` | 0-based iteration indices at which `f_*_list` snapshots were stored. |
| `NTK_kernal_fo` | First-order kernel matrix \(K_{\mathrm{NTK}} = XX^\top\) in feature space (samples × samples). |
| `NZK_kernal_zo` | Monte Carlo estimate of the NZK for the chosen \(z\) law (samples × samples). |

### Linear synthetic runs (`run_linear_ntk`)

Also includes: `theta_gt`, `data_sample` (unit-norm rows), `target_gt`, `noise_value`.

### Linear distribution / variance (`run_linear_ntk_distribution`, `run_linear_ntk_variance`)

- One FO run: same `theta_gt`, `data_sample`, `target_gt`, `noise_value`, `NTK_kernal_fo`, `loss_list_fo`, `f_fo_list`, etc.
- **Several ZO runs** are stored with **suffixes** on variable names, for example:
  - `loss_list_zo_normal_1.0`, `f_zo_list_t_10.0`, `NZK_kernal_zo_laplace_0.5`, `f_diff_list_zo_t_1000.0` (exact keys depend on `dist` and `scale`, or on variance strings like `0.5`).
- `run_linear_ntk_variance` also stores `variances` (the list used).

### FFN synthetic (`run_ffn_ntk`)

Includes `data_sample`, `target_gt`, and the common FO/ZO fields above (no `theta_gt` in the same form as linear; targets come from the synthetic linear teacher before the FFN features).

### FFN image runs (`run_ffn_mnist`, `run_ffn_cifar`, `run_ffn_imagenet`)

Same FO/ZO fields as above, plus:

| Variable | Meaning |
|----------|---------|
| `standard_loss_list_zo` | Per-iteration loss for **standard zeroth-order optimization in parameter space** (finite-difference along random directions), comparable to “ZO-parametric” curves in the paper/notebooks. |

**Plots:** This repository does not generate PDF figures automatically; use the arrays in `data.mat` to reproduce loss curves, kernel heatmaps, and function snapshots as in the original notebooks.

## Main hyperparameters (CLI)

| Flag | Role | Default (typical) |
|------|------|-------------------|
| `--eta_fo`, `--eta_zo` | Step sizes for FOGD and ZOGD | `1e-3` |
| `--iteration_total` | Number of iterations | `16000` |
| `--nzk_samples` | Monte Carlo samples to estimate NZK | `10000` |
| `--seed` | Random seed | script-dependent |
| `--device` | `cpu` or `cuda` | auto (`cuda` if available) |

Linear scripts also expose `--degree`, `--sample_num`; image scripts use `--resolution`, class indices, and `--sample_per_class`. See `--help` on each script.

## Correspondence to original notebooks

| Original folder | Refactored script / API |
|-----------------|-------------------------|
| **linear_zo_fo_ntk** | `run_linear_ntk.py`, `build_ntk`, `build_nzk_linear`, `train_fo_loop_with_f0`, `train_zo_loop_with_f0` |
| **linear_zo_fo_ntk_distribution** | `run_linear_ntk_distribution.py` |
| **linear_zo_fo_ntk_variance** | `run_linear_ntk_variance.py` |
| **FFN_zo_fo_ntk** | `run_ffn_ntk.py`, `FFN`, `linearize_ffn`, `build_nzk` |
| **FFN_zo_fo_mnist** | `run_ffn_mnist.py`, `train_zo_standard_loop` |
| **FFN_zo_fo_cifar** | `run_ffn_cifar.py` |
| **FFN_zo_fo_imagenet** | `run_ffn_imagenet.py` |

**Kernels:** NTK \(= XX^\top\). NZK uses \(\mathbb{E}_z[\cdot]\) with the same functional form as in the paper; linear scripts support normal / Student-\(t\) / Laplace and scales as in `build_nzk_linear`.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{zhang2026model,
  title     = {Model Evolution Under Zeroth-Order Optimization: A Neural Tangent Kernel Perspective},
  author    = {Chen Zhang and Yuxin Cheng and Chenchen Ding and Shuqi Wang and Jingreng Lei and Runsheng Yu and Yik-Chung WU and Ngai Wong},
  booktitle = {Workshop on Scientific Methods for Understanding Deep Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=PCJGU7DEEX},
}
```

**Paper link:** [https://openreview.net/forum?id=PCJGU7DEEX](https://openreview.net/forum?id=PCJGU7DEEX)

## License

[Specify your license, e.g. MIT, Apache-2.0.]
