import operator
import re
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy
from ..compat import cupy, has_cupy_gpu
def dish(X, *, inplace=False, threads_per_block=128, num_blocks=128):
    _is_float_array(X)
    out = X
    if not inplace:
        out = _alloc_like(X, zeros=False)
    if X.dtype == 'float32':
        dish_kernel_float((num_blocks,), (threads_per_block,), (out, X, X.size))
    else:
        dish_kernel_double((num_blocks,), (threads_per_block,), (out, X, X.size))
    return out