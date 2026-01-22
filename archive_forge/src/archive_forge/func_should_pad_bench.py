import functools
from itertools import chain
from typing import List, Optional
import torch
from torch import Tensor
from torch._inductor import utils
from torch.utils._mode_utils import no_dispatch
from torch.utils._triton import has_triton
from ..pattern_matcher import fwd_only, joint_fwd_bwd, Match, register_replacement
def should_pad_bench(mat1: Tensor, mat2: Tensor, op, input: Optional[Tensor]=None) -> bool:
    if not has_triton():
        return False
    do_bench = functools.partial(utils.do_bench, warmup=5)
    with no_dispatch():
        if op is torch.ops.aten.mm or op is torch.ops.aten.addmm:
            m = mat1.shape[0]
            k = mat1.shape[1]
            n = mat2.shape[1]
            m_padded_length = get_padded_length(m, get_alignment_size(mat1))
            k_padded_length = get_padded_length(k, get_alignment_size(mat1))
            n_padded_length = get_padded_length(n, get_alignment_size(mat2))
        elif op is torch.ops.aten.bmm:
            m = mat1.shape[1]
            k = mat2.shape[2]
            n = mat2.shape[2]
            m_padded_length = get_padded_length(m, get_alignment_size(mat1))
            k_padded_length = get_padded_length(k, get_alignment_size(mat1))
            n_padded_length = get_padded_length(n, get_alignment_size(mat2))
        else:
            return False
        if m_padded_length == k_padded_length == n_padded_length == 0:
            return False
        if not is_mm_compute_bound(m, k, n, mat1.dtype):
            return False
        key = should_pad_bench_key(mat1, mat2, op, input)
        cached_pad = get_cached_should_pad(key)
        if cached_pad is not None:
            return cached_pad
        mat1 = torch.randn_like(mat1)
        mat2 = torch.randn_like(mat2)
        if op is torch.ops.aten.bmm or op is torch.ops.aten.mm:
            ori_time = do_bench(lambda: op(mat1, mat2))
        else:
            if input is not None:
                input = torch.randn_like(input)
            ori_time = do_bench(lambda: op(input, mat1, mat2))
        mat1_pad = torch.randn_like(mat1)
        mat2_pad = torch.randn_like(mat2)
        if op is torch.ops.aten.addmm:
            input_pad = None
            if input is not None and input.is_cuda:
                input_pad = torch.randn_like(input)
            pad_time = do_bench(lambda: pad_addmm(input_pad, mat1_pad, mat2_pad, m_padded_length, k_padded_length, n_padded_length))
        elif op is torch.ops.aten.mm:
            pad_time = do_bench(lambda: pad_mm(mat1_pad, mat2_pad, m_padded_length, k_padded_length, n_padded_length))
        else:
            pad_time = do_bench(lambda: pad_bmm(mat1_pad, mat2_pad, m_padded_length, k_padded_length, n_padded_length))
        should_pad = ori_time > pad_time * 1.1
        set_cached_should_pad(key, should_pad)
        return should_pad