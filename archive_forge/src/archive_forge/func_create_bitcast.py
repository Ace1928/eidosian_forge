import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def create_bitcast(self, src, dst_type):
    return TensorHandle(src.data.view(self.np_dtype(dst_type)), dst_type)