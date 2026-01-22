import functools
import torch
from ..lowering import lowerings
from ..select_algorithm import (
from ..utils import use_aten_gemm_kernels, use_triton_template
from ..virtualized import V
from .mm_common import mm_args, mm_grid, mm_options
@functools.lru_cache(None)
def mm_configs():
    import triton
    mm_triton_configs = [{'config': {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, 'num_stages': 2, 'num_warps': 4, 'cond': True}, {'config': {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, 'num_stages': 3, 'num_warps': 8, 'cond': True}, {'config': {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, 'num_stages': 4, 'num_warps': 16, 'cond': True}, {'config': {'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, 'num_stages': 4, 'num_warps': 8, 'cond': True}, {'config': {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, 'num_stages': 4, 'num_warps': 8, 'cond': True}, {'config': {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, 'num_stages': 1, 'num_warps': 8, 'cond': True}, {'config': {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, 'num_stages': 1, 'num_warps': 8, 'cond': True}, {'config': {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 128}, 'num_stages': 1, 'num_warps': 8, 'cond': torch.version.hip is None}, {'config': {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, 'num_stages': 2, 'num_warps': 4, 'cond': True}, {'config': {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, 'num_stages': 1, 'num_warps': 2, 'cond': True}]
    if torch.version.hip:
        filtered_configs = [triton.Config(c['config'], num_stages=1, num_warps=c['num_warps']) for c in mm_triton_configs if c['cond']]
    else:
        filtered_configs = [triton.Config(c['config'], num_stages=c['num_stages'], num_warps=c['num_warps']) for c in mm_triton_configs if c['cond']]
    return filtered_configs