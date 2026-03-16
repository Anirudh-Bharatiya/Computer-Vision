# src/utils.py
"""
Small utilities used across scripts.

- SEED is fixed to the roll number per assignment.
- get_default_num_workers recommends a safe default.
"""
import multiprocessing
import platform
import torch
import random
import numpy as np

SEED = 2023090

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_default_num_workers(device: torch.device = None) -> int:
    """
    Recommend safe num_workers:
      - On Darwin/MPS: 0
      - Else: min(4, cpu_count//2)
    """
    cpus = multiprocessing.cpu_count() or 1
    sysname = platform.system().lower()
    if device is not None:
        if device.type == 'mps' or sysname == 'darwin':
            return 0
    else:
        if sysname == 'darwin':
            return 0
    return max(0, min(4, max(1, cpus // 2)))