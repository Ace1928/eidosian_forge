from collections import namedtuple
import itertools
import functools
import operator
import ctypes
import numpy as np
from numba import _helperlib
from numba.core import config
def iter_strides_c_contig(arr, shape=None):
    """yields the c-contiguous strides
    """
    shape = arr.shape if shape is None else shape
    itemsize = arr.itemsize

    def gen():
        yield itemsize
        sum = 1
        for s in reversed(shape[1:]):
            sum *= s
            yield (sum * itemsize)
    for i in reversed(list(gen())):
        yield i