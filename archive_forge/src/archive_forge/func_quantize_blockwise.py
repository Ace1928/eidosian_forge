import ctypes as ct
from functools import reduce  # Required in Python 3
import itertools
import operator
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from bitsandbytes.utils import pack_dict_to_tensor, unpack_tensor_to_dict
from .cextension import COMPILED_WITH_CUDA, lib
def quantize_blockwise(A: Tensor, code: Optional[torch.Tensor]=None, absmax: Optional[torch.Tensor]=None, out: Optional[torch.Tensor]=None, blocksize=4096, nested=False) -> Tuple[Tensor, QuantState]:
    """
    Quantize tensor A in blocks of size 4096 values.

    Quantizes tensor A by dividing it into blocks of 4096 values.
    Then the absolute maximum value within these blocks is calculated
    for the non-linear quantization.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor.
    code : torch.Tensor
        The quantization map.
    absmax : torch.Tensor
        The absmax values.
    out : torch.Tensor
        The output tensor (8-bit).

    Returns
    -------
    torch.Tensor:
        The 8-bit tensor.
    tuple(torch.Tensor, torch.Tensor):
        The quantization state to undo the quantization.
    """
    if code is None:
        if 'dynamic' not in name2qmap:
            name2qmap['dynamic'] = create_dynamic_map().to(A.device)
        code = name2qmap['dynamic']
    if absmax is None:
        n = A.numel()
        blocks = n // blocksize
        blocks += 1 if n % blocksize > 0 else 0
        absmax = torch.zeros((blocks,), device=A.device, dtype=torch.float32)
    if out is None:
        out = torch.zeros_like(A, dtype=torch.uint8)
    if A.device.type != 'cpu':
        assert blocksize in [4096, 2048, 1024, 512, 256, 128, 64]
        cblocksize = ct.c_int32(blocksize)
        prev_device = pre_call(A.device)
        code = code.to(A.device)
        is_on_gpu([code, A, out, absmax])
        if A.dtype == torch.float32:
            lib.cquantize_blockwise_fp32(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), cblocksize, ct.c_int(A.numel()))
        elif A.dtype == torch.float16:
            lib.cquantize_blockwise_fp16(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), cblocksize, ct.c_int(A.numel()))
        elif A.dtype == torch.bfloat16:
            lib.cquantize_blockwise_bf16(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), cblocksize, ct.c_int(A.numel()))
        else:
            raise ValueError(f'Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}')
        post_call(A.device)
    else:
        code = code.cpu()
        lib.cquantize_blockwise_cpu_fp32(get_ptr(code), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_longlong(blocksize), ct.c_longlong(A.numel()))
    if nested:
        offset = absmax.mean()
        absmax -= offset
        qabsmax, state2 = quantize_blockwise(absmax, blocksize=blocksize, nested=False)
        quant_state = QuantState(absmax=qabsmax, code=code, blocksize=blocksize, dtype=A.dtype, offset=offset, state2=state2)
    else:
        quant_state = QuantState(absmax=absmax, code=code, blocksize=blocksize, dtype=A.dtype)
    return (out, quant_state)