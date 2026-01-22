import operator
import re
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy
from ..compat import cupy, has_cupy_gpu
def backprop_swish(dY, X, Y, *, inplace=False, threshold=17.0, threads_per_block=128, num_blocks=128):
    _is_float_array(dY)
    _is_float_array(X, shape=dY.shape)
    _is_float_array(Y, shape=dY.shape)
    out = dY
    if not inplace:
        out = _alloc_like(dY, zeros=False)
    if dY.dtype == 'float32':
        backprop_swish_kernel_float((num_blocks,), (threads_per_block,), (out, dY, X, Y, threshold, out.size))
    else:
        backprop_swish_kernel_double((num_blocks,), (threads_per_block,), (out, dY, X, Y, threshold, out.size))
    return out