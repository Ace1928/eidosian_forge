import math
import os
import torch
import weakref
from functools import lru_cache
from torch.utils._triton import has_triton
from ._triton_ops_meta import get_meta
from typing import Optional, Tuple
def bsr_softmax(input, max_row_nnz=None):
    f_name = 'bsr_softmax'
    check_bsr_layout(f_name, input)
    check_dtype(f_name, input, input.dtype)
    if input._nnz() == 0 or input.numel() == 0:
        return input.clone()
    m, n = input.shape[-2:]
    nnz = input._nnz()
    row_block, col_block = input.values().shape[-2:]
    if max_row_nnz is None:
        max_row_nnz = triton.next_power_of_2(n)
    else:
        max_row_nnz = triton.next_power_of_2(max_row_nnz)
    crow_indices = input.crow_indices().unsqueeze(0).flatten(0, -2)
    if input.values().transpose(-3, -2).is_contiguous():
        values = input.values().clone()
    else:
        values = input.values()
    values = values.transpose(-3, -2).contiguous().unsqueeze(0).flatten(0, -4).reshape(-1, row_block, nnz * col_block)
    full_grid = (values.shape[0], row_block, m // row_block)
    grid_blocks = None
    tensor_dims_map = {crow_indices[..., :-1]: (0, None, -1), values: (0, None, None)}

    def kernel(grid, *sliced_tensors):
        _bsr_softmax_kernel[grid](*ptr_stride_extractor(*sliced_tensors), row_block, col_block, max_row_nnz, min(2 ** 17, max_row_nnz))
    launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks)
    values = values.reshape(-1, row_block, nnz, col_block).transpose(-3, -2).reshape(*input.values().shape)
    return torch.sparse_compressed_tensor(input.crow_indices().clone(), input.col_indices().clone(), values, size=input.shape, layout=input.layout)