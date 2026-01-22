import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def get_block_ty(self, dtype, shape):
    return tl.tensor(shape, dtype)