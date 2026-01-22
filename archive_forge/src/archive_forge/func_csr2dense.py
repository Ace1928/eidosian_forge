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
def csr2dense(a, order):
    out = cupy.zeros(a.shape, dtype=a.dtype, order=order)
    m, n = a.shape
    cupy_csr2dense()(m, n, a.indptr, a.indices, a.data, order == 'C', out)
    return out