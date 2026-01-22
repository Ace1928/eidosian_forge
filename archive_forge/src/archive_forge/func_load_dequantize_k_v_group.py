import functools
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type
import torch
from ..common import _has_triton21, register_operator
from .attn_bias import (
from .common import AttentionFwOpBase, Context, Inputs, check_lastdim_alignment_stride1
@triton.jit
def load_dequantize_k_v_group(K_block_ptr, V_block_ptr, K_scale_shift_block_ptr, V_scale_shift_block_ptr, BOUNDS_CHECKS_N: tl.constexpr, PACKED_PER_VAL: tl.constexpr, PACKED_D_PER_GROUP: tl.constexpr, dtype: tl.constexpr, group_id: tl.constexpr):
    """Load K/V for a given block. In case of int4-quantized K/V, dequantize them after loading.
        If quantization is group-wise, use group_id to advance the pointers to the current group.
        """
    K_block_ptr = tl.advance(K_block_ptr, (PACKED_D_PER_GROUP * group_id, 0))
    V_block_ptr = tl.advance(V_block_ptr, (0, PACKED_D_PER_GROUP * group_id))
    k = tl.load(K_block_ptr, boundary_check=(1,) if BOUNDS_CHECKS_N else ())
    v = tl.load(V_block_ptr, boundary_check=(0,) if BOUNDS_CHECKS_N else ())
    if PACKED_PER_VAL > 1:
        K_scale_shift_block_ptr = tl.advance(K_scale_shift_block_ptr, (group_id, 0))
        V_scale_shift_block_ptr = tl.advance(V_scale_shift_block_ptr, (0, group_id))
        k_scale_shift = tl.load(K_scale_shift_block_ptr, boundary_check=(1,) if BOUNDS_CHECKS_N else ())
        v_scale_shift = tl.load(V_scale_shift_block_ptr, boundary_check=(0,) if BOUNDS_CHECKS_N else ())
        k_scale, k_shift = cast_uint32_to_half2(k_scale_shift)
        v_scale, v_shift = cast_uint32_to_half2(v_scale_shift)
        v = dequantize(v, v_scale, v_shift, PACKED_PER_VAL).to(dtype)
        k_t = dequantize(tl.trans(k), tl.trans(k_scale), tl.trans(k_shift), PACKED_PER_VAL).to(dtype)
        k = tl.trans(k_t)
    return (k, v)