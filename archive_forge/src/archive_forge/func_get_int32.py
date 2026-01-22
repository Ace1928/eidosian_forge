import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def get_int32(self, value):
    return TensorHandle(np.array([value], dtype=np.int32), tl.int32)