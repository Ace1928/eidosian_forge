from dataclasses import dataclass
from functools import reduce  # Required in Python 3
import operator
from typing import Callable, Optional, Tuple
import warnings
from warnings import warn
import torch
import bitsandbytes.functional as F
def matmul_4bit(A: torch.Tensor, B: torch.Tensor, quant_state: F.QuantState, out: Optional[torch.Tensor]=None, bias=None):
    assert quant_state is not None
    if A.numel() == A.shape[-1] and A.requires_grad == False:
        if A.shape[-1] % quant_state.blocksize != 0:
            warn(f'Some matrices hidden dimension is not a multiple of {quant_state.blocksize} and efficient inference kernels are not supported for these (slow). Matrix input size found: {A.shape}')
            return MatMul4Bit.apply(A, B, out, bias, quant_state)
        else:
            out = F.gemv_4bit(A, B.t(), out, state=quant_state)
            if bias is not None:
                out += bias
            return out
    else:
        return MatMul4Bit.apply(A, B, out, bias, quant_state)