import math
from pathlib import Path
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.cpp_extension import load
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_mra import MraConfig
def sparse_max(sparse_qk_prod, indices, query_num_block, key_num_block):
    """
    Computes maximum values for softmax stability.
    """
    if len(sparse_qk_prod.size()) != 4:
        raise ValueError('sparse_qk_prod must be a 4-dimensional tensor.')
    if len(indices.size()) != 2:
        raise ValueError('indices must be a 2-dimensional tensor.')
    if sparse_qk_prod.size(2) != 32:
        raise ValueError('The size of the second dimension of sparse_qk_prod must be 32.')
    if sparse_qk_prod.size(3) != 32:
        raise ValueError('The size of the third dimension of sparse_qk_prod must be 32.')
    index_vals = sparse_qk_prod.max(dim=-2).values.transpose(-1, -2)
    index_vals = index_vals.contiguous()
    indices = indices.int()
    indices = indices.contiguous()
    max_vals, max_vals_scatter = mra_cuda_kernel.index_max(index_vals, indices, query_num_block, key_num_block)
    max_vals_scatter = max_vals_scatter.transpose(-1, -2)[:, :, None, :]
    return (max_vals, max_vals_scatter)