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
def mm_to_sparse(dense_query, dense_key, indices, block_size=32):
    """
    Performs Sampled Dense Matrix Multiplication.
    """
    batch_size, query_size, dim = dense_query.size()
    _, key_size, dim = dense_key.size()
    if query_size % block_size != 0:
        raise ValueError('query_size (size of first dimension of dense_query) must be divisible by block_size.')
    if key_size % block_size != 0:
        raise ValueError('key_size (size of first dimension of dense_key) must be divisible by block_size.')
    dense_query = dense_query.reshape(batch_size, query_size // block_size, block_size, dim).transpose(-1, -2)
    dense_key = dense_key.reshape(batch_size, key_size // block_size, block_size, dim).transpose(-1, -2)
    if len(dense_query.size()) != 4:
        raise ValueError('dense_query must be a 4-dimensional tensor.')
    if len(dense_key.size()) != 4:
        raise ValueError('dense_key must be a 4-dimensional tensor.')
    if len(indices.size()) != 2:
        raise ValueError('indices must be a 2-dimensional tensor.')
    if dense_query.size(3) != 32:
        raise ValueError('The third dimension of dense_query must be 32.')
    if dense_key.size(3) != 32:
        raise ValueError('The third dimension of dense_key must be 32.')
    dense_query = dense_query.contiguous()
    dense_key = dense_key.contiguous()
    indices = indices.int()
    indices = indices.contiguous()
    return mra_cuda_kernel.mm_to_sparse(dense_query, dense_key, indices.int())