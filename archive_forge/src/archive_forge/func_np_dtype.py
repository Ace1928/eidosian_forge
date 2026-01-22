import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def np_dtype(self, tt_dtype):
    if isinstance(tt_dtype, tl.pointer_type):
        return np.dtype(np.uint64)
    np_types = {tl.float16: np.dtype(np.float16), tl.float32: np.dtype(np.float32), tl.float64: np.dtype(np.float64), tl.int8: np.dtype(np.int8), tl.uint8: np.dtype(np.uint8), tl.int16: np.dtype(np.int16), tl.uint16: np.dtype(np.uint16), tl.int32: np.dtype(np.int32), tl.uint32: np.dtype(np.uint32), tl.int64: np.dtype(np.int64), tl.uint64: np.dtype(np.uint64)}
    return np_types[tt_dtype]