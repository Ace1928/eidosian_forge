import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def create_splat(self, arg, shape):
    return TensorHandle(np.full(shape, arg.data[0], dtype=self.np_dtype(arg.dtype)), arg.dtype)