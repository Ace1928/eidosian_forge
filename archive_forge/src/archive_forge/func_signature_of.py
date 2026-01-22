from typing import Dict, List, Union
import torch
from .. import config
from ..utils import instance_descriptor
from ..virtualized import V
from .common import SizeArg, TensorArg
def signature_of(arg: Union[TensorArg, SizeArg], *, size_dtype: str) -> str:
    from triton.runtime.jit import JITFunction
    if isinstance(arg, TensorArg):
        if arg.dtype == torch.float8_e4m3fn:
            tye = '*fp8e4nv'
        elif arg.dtype == torch.float8_e5m2:
            tye = '*fp8e5'
        else:
            tye = JITFunction._type_of(arg.dtype)
        if V.graph.is_unspec_arg(arg.buffer):
            new_tye = tye.lstrip('*')
            if new_tye in ['fp16', 'bf16']:
                return 'fp32'
            else:
                return new_tye
        else:
            return tye
    if isinstance(arg, SizeArg):
        if arg.expr is None:
            return '*i8'
        elif isinstance(arg.expr, float):
            return 'fp32'
        if size_dtype == 'tl.int32':
            return 'i32'
        elif size_dtype == 'tl.int64':
            return 'i64'
        else:
            raise NotImplementedError(f'unhandled size_dtype {size_dtype}')
    raise NotImplementedError(f'unhandled {type(arg)}: {arg}')