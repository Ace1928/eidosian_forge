import operator
import re
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy
from ..compat import cupy, has_cupy_gpu
def backprop_seq2col(dY, nW, *, lengths=None, threads_per_block=128, num_blocks=128):
    _is_float_array(dY)
    B = dY.shape[0]
    nF = nW * 2 + 1
    I = dY.shape[1] // nF
    lengths = check_seq2col_lengths(lengths, B)
    nL = lengths.shape[0]
    out = _alloc((B, I), dtype=dY.dtype, zeros=True)
    if dY.size != 0 and lengths.size != 0:
        if dY.dtype == 'float32':
            backprop_seq2col_kernel_float((num_blocks,), (threads_per_block,), (out, dY, lengths, nW, B, I, nL))
        else:
            backprop_seq2col_kernel_double((num_blocks,), (threads_per_block,), (out, dY, lengths, nW, B, I, nL))
    return out