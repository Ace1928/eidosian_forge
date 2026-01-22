from collections import defaultdict
import logging
import math
from typing import Dict
import torch
import torch.distributed as dist
from . import default_hooks as default
from torch.distributed import distributed_c10d
def create_low_rank_tensor(fill_random_values, rng):
    """Returns a low-rank 2D tensor of square_side_length * matrix_approximation_rank."""
    if fill_random_values:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(rng.randint(1000000000))
            return torch.randn(square_side_length, state.matrix_approximation_rank, device='cpu', dtype=input_tensor.dtype).to(device)
    else:
        return torch.empty(square_side_length, state.matrix_approximation_rank, device=device, dtype=input_tensor.dtype)