from collections import namedtuple
import itertools
import functools
import operator
import ctypes
import numpy as np
from numba import _helperlib
from numba.core import config
@classmethod
def from_desc(cls, offset, shape, strides, itemsize):
    dims = []
    for ashape, astride in zip(shape, strides):
        dim = Dim(offset, offset + ashape * astride, ashape, astride, single=False)
        dims.append(dim)
        offset = 0
    return cls(dims, itemsize)