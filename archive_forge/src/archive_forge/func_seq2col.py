import operator
import re
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy
from ..compat import cupy, has_cupy_gpu
def seq2col(seq, nW, *, lengths=None, threads_per_block=128, num_blocks=128):
    _is_float_array(seq)
    B = seq.shape[0]
    nF = nW * 2 + 1
    I = seq.shape[1]
    lengths = check_seq2col_lengths(lengths, B)
    nL = lengths.shape[0]
    out = _alloc((B, I * nF), dtype=seq.dtype, zeros=True)
    if seq.size != 0 and lengths.size != 0:
        if seq.dtype == 'float32':
            seq2col_kernel_float((num_blocks,), (threads_per_block,), (out, seq, lengths, nW, B, I, nL))
        else:
            seq2col_kernel_double((num_blocks,), (threads_per_block,), (out, seq, lengths, nW, B, I, nL))
    return out