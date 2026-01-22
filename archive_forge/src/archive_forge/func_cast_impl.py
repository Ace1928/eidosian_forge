import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def cast_impl(self, src, dst_type):
    if isinstance(dst_type, tl.tensor):
        dst_type = dst_type.dtype
    return TensorHandle(src.data.astype(self.np_dtype(dst_type)), dst_type)