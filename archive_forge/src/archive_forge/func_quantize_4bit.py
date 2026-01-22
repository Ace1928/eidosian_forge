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
def quantize_4bit(A: Tensor, absmax: Optional[torch.Tensor]=None, out: Optional[torch.Tensor]=None, blocksize=64, compress_statistics=False, quant_type='fp4', quant_storage=torch.uint8) -> Tuple[Tensor, QuantState]:
    """
    Quantize tensor A in blocks of 4-bit values.

    Quantizes tensor A by dividing it into blocks which are independently quantized to FP4.

    Parameters
    ----------
    A : torch.Tensor
        The input tensor.
    absmax : torch.Tensor
        The absmax values.
    out : torch.Tensor
        The output tensor.
    blocksize : int
        The blocksize used in quantization.
    quant_type : str
        The 4-bit quantization data type {fp4, nf4}

    Returns
    -------
    torch.Tensor:
        Tensor with packed 4-bit values.
    tuple(torch.Tensor, torch.Size, torch.dtype, int):
        The quantization state to undo the quantization.
    """
    if A.device.type != 'cuda':
        raise NotImplementedError(f'Device type not supported for FP4 quantization: {A.device.type}')
    if quant_type not in ['fp4', 'nf4']:
        raise NotImplementedError(f'4-bit quantization data type {quant_type} is not implemented.')
    n = A.numel()
    input_shape = A.shape
    if absmax is None:
        blocks = n // blocksize
        blocks += 1 if n % blocksize > 0 else 0
        absmax = torch.zeros((blocks,), device=A.device, dtype=torch.float32)
    if out is None:
        mod = dtype2bytes[quant_storage] * 2
        out = torch.zeros(((n + 1) // mod, 1), dtype=quant_storage, device=A.device)
    assert blocksize in [4096, 2048, 1024, 512, 256, 128, 64]
    prev_device = pre_call(A.device)
    is_on_gpu([A, out, absmax])
    if A.dtype == torch.float32:
        if quant_type == 'fp4':
            lib.cquantize_blockwise_fp32_fp4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int32(blocksize), ct.c_int(n))
        else:
            lib.cquantize_blockwise_fp32_nf4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int32(blocksize), ct.c_int(n))
    elif A.dtype == torch.float16:
        if quant_type == 'fp4':
            lib.cquantize_blockwise_fp16_fp4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int32(blocksize), ct.c_int(n))
        else:
            lib.cquantize_blockwise_fp16_nf4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int32(blocksize), ct.c_int(n))
    elif A.dtype == torch.bfloat16:
        if quant_type == 'fp4':
            lib.cquantize_blockwise_bf16_fp4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int32(blocksize), ct.c_int(n))
        else:
            lib.cquantize_blockwise_bf16_nf4(get_ptr(None), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int32(blocksize), ct.c_int(n))
    else:
        raise ValueError(f'Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}')
    post_call(A.device)
    code = get_4bit_type(quant_type, device=A.device)
    if compress_statistics:
        offset = absmax.mean()
        absmax -= offset
        qabsmax, state2 = quantize_blockwise(absmax, blocksize=256)
        del absmax
        state = QuantState(absmax=qabsmax, shape=input_shape, dtype=A.dtype, blocksize=blocksize, code=code, quant_type=quant_type, offset=offset, state2=state2)
    else:
        state = QuantState(absmax=absmax, shape=input_shape, dtype=A.dtype, blocksize=blocksize, code=code, quant_type=quant_type)
    return (out, state)