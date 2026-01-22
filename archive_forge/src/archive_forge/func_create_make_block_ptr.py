import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def create_make_block_ptr(self, base, shape, strides, offsets, tensor_shape, order):
    return BlockPointerHandle(base, shape, strides, np.array(offsets), tensor_shape, order)