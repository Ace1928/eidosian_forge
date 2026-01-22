import functools as _functools
import numpy as _numpy
import platform as _platform
import cupy as _cupy
from cupy_backends.cuda.api import driver as _driver
from cupy_backends.cuda.api import runtime as _runtime
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupy._core import _dtype
from cupy.cuda import device as _device
from cupy.cuda import stream as _stream
from cupy import _util
import cupyx.scipy.sparse
def csr2csr_compress(x, tol):
    if not check_availability('csr2csr_compress'):
        raise RuntimeError('csr2csr_compress is not available.')
    assert x.dtype.char in 'fdFD'
    handle = _device.get_cusparse_handle()
    m, n = x.shape
    nnz_per_row = _cupy.empty(m, 'i')
    nnz = _call_cusparse('nnz_compress', x.dtype, handle, m, x._descr.descriptor, x.data.data.ptr, x.indptr.data.ptr, nnz_per_row.data.ptr, tol)
    data = _cupy.zeros(nnz, x.dtype)
    indptr = _cupy.empty(m + 1, 'i')
    indices = _cupy.zeros(nnz, 'i')
    _call_cusparse('csr2csr_compress', x.dtype, handle, m, n, x._descr.descriptor, x.data.data.ptr, x.indices.data.ptr, x.indptr.data.ptr, x.nnz, nnz_per_row.data.ptr, data.data.ptr, indices.data.ptr, indptr.data.ptr, tol)
    return cupyx.scipy.sparse.csr_matrix((data, indices, indptr), shape=x.shape)