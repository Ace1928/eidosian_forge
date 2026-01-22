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
def dequantize_blockwise(A: Tensor, quant_state: Optional[QuantState]=None, absmax: Optional[torch.Tensor]=None, code: Optional[torch.Tensor]=None, out: Optional[torch.Tensor]=None, blocksize: int=4096, nested=False) -> Tensor:
    """
    Dequantizes blockwise quantized values.

    Dequantizes the tensor A with maximum absolute values absmax in
    blocks of size 4096.

    Parameters
    ----------
    A : torch.Tensor
        The input 8-bit tensor.
    quant_state : QuantState
        Object with code, absmax and other quantization state components.
    absmax : torch.Tensor
        The absmax values.
    code : torch.Tensor
        The quantization map.
    out : torch.Tensor
        Dequantized output tensor (default: float32)


    Returns
    -------
    torch.Tensor:
        Dequantized tensor (default: float32)
    """
    assert quant_state is not None or absmax is not None
    if code is None and quant_state is None:
        if 'dynamic' not in name2qmap:
            name2qmap['dynamic'] = create_dynamic_map().to(A.device)
        code = name2qmap['dynamic']
    if quant_state is None:
        quant_state = QuantState(absmax=absmax, code=code, blocksize=blocksize, dtype=torch.float32)
    absmax = quant_state.absmax
    if quant_state.nested:
        absmax = dequantize_blockwise(quant_state.absmax, quant_state.state2)
        absmax += quant_state.offset
        if absmax.dtype != torch.float32:
            absmax = absmax.float()
    if out is None:
        out = torch.empty(A.shape, dtype=quant_state.dtype, device=A.device)
    if A.device.type != 'cpu':
        device = pre_call(A.device)
        code = quant_state.code.to(A.device)
        if quant_state.blocksize not in [2048, 4096, 1024, 512, 256, 128, 64]:
            raise ValueError(f'The blockwise of {quant_state.blocksize} is not supported. Supported values: [2048, 4096, 1024, 512, 256, 128, 64]')
        is_on_gpu([A, absmax, out])
        if out.dtype == torch.float32:
            lib.cdequantize_blockwise_fp32(get_ptr(quant_state.code), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(quant_state.blocksize), ct.c_int(A.numel()))
        elif out.dtype == torch.float16:
            lib.cdequantize_blockwise_fp16(get_ptr(quant_state.code), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(quant_state.blocksize), ct.c_int(A.numel()))
        elif out.dtype == torch.bfloat16:
            lib.cdequantize_blockwise_bf16(get_ptr(quant_state.code), get_ptr(A), get_ptr(absmax), get_ptr(out), ct.c_int(quant_state.blocksize), ct.c_int(A.numel()))
        else:
            raise ValueError(f'Blockwise quantization only supports 16/32-bit floats, but got {A.dtype}')
        post_call(A.device)
    else:
        code = quant_state.code.cpu()
        lib.cdequantize_blockwise_cpu_fp32(get_ptr(code), get_ptr(A), get_ptr(quant_state.absmax), get_ptr(out), ct.c_longlong(quant_state.blocksize), ct.c_longlong(A.numel()))
    return out