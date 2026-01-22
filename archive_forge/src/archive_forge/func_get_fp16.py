import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def get_fp16(self, value):
    return TensorHandle(np.array([value], dtype=np.float16), tl.float16)