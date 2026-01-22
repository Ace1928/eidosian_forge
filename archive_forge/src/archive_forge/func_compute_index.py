from collections import namedtuple
import itertools
import functools
import operator
import ctypes
import numpy as np
from numba import _helperlib
from numba.core import config
def compute_index(indices, dims):
    return sum((d.get_offset(i) for i, d in zip(indices, dims)))