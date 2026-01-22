import operator
import re
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy
from ..compat import cupy, has_cupy_gpu
def backprop_maxout(dY, which, P, *, threads_per_block=128, num_blocks=128):
    _is_float_array(dY)
    B = dY.shape[0]
    I = dY.shape[1]
    out = _alloc((B, I, P), dtype=dY.dtype, zeros=True)
    _check_which_maxout(which, B, I, P)
    if dY.dtype == 'float32':
        backprop_maxout_kernel_float((num_blocks,), (threads_per_block,), (out, dY, which, B, I, P))
    else:
        backprop_maxout_kernel_double((num_blocks,), (threads_per_block,), (out, dY, which, B, I, P))
    return out