import math
from typing import List
import numpy as np
import torch
from xformers.components.attention.sparsity_config import (
def random_pattern_from_probability_matrix(dist_matrix, nnz):
    att = torch.zeros_like(dist_matrix, dtype=torch.bool)
    if dist_matrix.numel() > 2 ** 24:
        dist_matrix = dist_matrix.double()
        dist_matrix /= dist_matrix.sum()
        idxs = np.random.choice(dist_matrix.numel(), nnz, p=dist_matrix.flatten(), replace=False)
        idxs = torch.as_tensor(idxs)
    else:
        idxs = torch.multinomial(dist_matrix.flatten(), nnz, replacement=False)
    att.view(-1)[idxs] = True
    return att