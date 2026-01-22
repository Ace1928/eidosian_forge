import operator
import re
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy
from ..compat import cupy, has_cupy_gpu
def maxout(X, *, threads_per_block=128, num_blocks=128):
    _is_float_array(X)
    B, I, P = X.shape
    out_shape = (B, I)
    best = _alloc(out_shape, dtype=X.dtype, zeros=False)
    which = _alloc(out_shape, dtype='i', zeros=False)
    if X.dtype == 'float32':
        maxout_kernel_float((num_blocks,), (threads_per_block,), (best, which, X, B, I, P))
    else:
        maxout_kernel_double((num_blocks,), (threads_per_block,), (best, which, X, B, I, P))
    return (best, which)