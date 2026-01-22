import itertools
import warnings
from . import core as ma
from .core import (
import numpy as np
from numpy import ndarray, array as nxarray
from numpy.core.multiarray import normalize_axis_index
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.function_base import _ureduce
from numpy.lib.index_tricks import AxisConcatenator
def flatten_inplace(seq):
    """Flatten a sequence in place."""
    k = 0
    while k != len(seq):
        while hasattr(seq[k], '__iter__'):
            seq[k:k + 1] = seq[k]
        k += 1
    return seq