"""
Utilities: save results to .mat, optional plotting.
"""

import os
from typing import Any, Dict
import torch
from scipy.io import savemat


def save_results(output_dir: str, data: Dict[str, Any]) -> None:
    """Save dict of arrays/tensors to output_dir/data.mat (numpy arrays)."""
    os.makedirs(output_dir, exist_ok=True)
    out = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy()
        elif isinstance(v, list):
            out[k] = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in v]
        else:
            out[k] = v
    savemat(os.path.join(output_dir, "data.mat"), out)
