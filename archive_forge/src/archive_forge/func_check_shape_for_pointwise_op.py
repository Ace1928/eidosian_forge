import operator
import warnings
import numpy
import cupy
from cupy._core import _accelerator
from cupy.cuda import cub
from cupy.cuda import runtime
from cupyx import cusparse
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _compressed
from cupyx.scipy.sparse import _csc
from cupyx.scipy.sparse import SparseEfficiencyWarning
from cupyx.scipy.sparse import _util
def check_shape_for_pointwise_op(a_shape, b_shape, allow_broadcasting=True):
    if allow_broadcasting:
        a_m, a_n = a_shape
        b_m, b_n = b_shape
        if not (a_m == b_m or a_m == 1 or b_m == 1):
            raise ValueError('inconsistent shape')
        if not (a_n == b_n or a_n == 1 or b_n == 1):
            raise ValueError('inconsistent shape')
    elif a_shape != b_shape:
        raise ValueError('inconsistent shape')