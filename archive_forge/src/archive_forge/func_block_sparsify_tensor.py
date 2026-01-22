import math
from typing import List
import numpy as np
import torch
from xformers.components.attention.sparsity_config import (
def block_sparsify_tensor(x, mask, block_size):
    """
    Block sparsify a tensor, given a mask and block size
    """
    ret = torch.empty((x.size(0), mask.sum(), block_size, block_size), dtype=x.dtype, device=x.device)
    for idx, (h, i, j) in enumerate(zip(*mask.nonzero(as_tuple=True))):
        ret[:, idx, :, :] = x[:, h, i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
    return ret