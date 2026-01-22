from collections import namedtuple
import itertools
import functools
import operator
import ctypes
import numpy as np
from numba import _helperlib
from numba.core import config
def iter_strides_f_contig(arr, shape=None):
    """yields the f-contiguous strides
    """
    shape = arr.shape if shape is None else shape
    itemsize = arr.itemsize
    yield itemsize
    sum = 1
    for s in shape[:-1]:
        sum *= s
        yield (sum * itemsize)