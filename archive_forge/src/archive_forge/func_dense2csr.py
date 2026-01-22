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
def dense2csr(a):
    if a.dtype.char in 'fdFD':
        if cusparse.check_availability('denseToSparse'):
            return cusparse.denseToSparse(a, format='csr')
        else:
            return cusparse.dense2csr(a)
    m, n = a.shape
    a = cupy.ascontiguousarray(a)
    indptr = cupy.zeros(m + 1, dtype=numpy.int32)
    info = cupy.zeros(m * n + 1, dtype=numpy.int32)
    cupy_dense2csr_step1()(m, n, a, indptr, info)
    indptr = cupy.cumsum(indptr, dtype=numpy.int32)
    info = cupy.cumsum(info, dtype=numpy.int32)
    nnz = int(indptr[-1])
    indices = cupy.empty(nnz, dtype=numpy.int32)
    data = cupy.empty(nnz, dtype=a.dtype)
    cupy_dense2csr_step2()(m, n, a, info, indices, data)
    return csr_matrix((data, indices, indptr), shape=(m, n))