import warnings
import numpy
import cupy
from cupy import _core
from cupy import _util
def do_map(inputs, output):
    """labels must be sorted"""
    nidx = sorted_index.size
    lo = cupy.searchsorted(labels, sorted_index, side='left')
    hi = cupy.searchsorted(labels, sorted_index, side='right')
    for i, low, high in zip(range(nidx), lo, hi):
        if low == high:
            continue
        output[i] = func(*[inp[low:high] for inp in inputs])