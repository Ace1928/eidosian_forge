import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def create_int_to_ptr(self, val, dst_ty):
    return TensorHandle(val.data.astype(np.uint64), dst_ty)