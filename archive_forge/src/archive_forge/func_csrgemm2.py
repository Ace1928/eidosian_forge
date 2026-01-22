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
def csrgemm2(a, b, d=None, alpha=1, beta=1):
    """Matrix-matrix product for CSR-matrix.

    math::
       C = alpha * A * B + beta * D

    Args:
        a (cupyx.scipy.sparse.csr_matrix): Sparse matrix A.
        b (cupyx.scipy.sparse.csr_matrix): Sparse matrix B.
        d (cupyx.scipy.sparse.csr_matrix or None): Sparse matrix D.
        alpha (scalar): Coefficient
        beta (scalar): Coefficient

    Returns:
        cupyx.scipy.sparse.csr_matrix

    """
    if not check_availability('csrgemm2'):
        raise RuntimeError('csrgemm2 is not available.')
    assert a.ndim == b.ndim == 2
    if not isinstance(a, cupyx.scipy.sparse.csr_matrix):
        raise TypeError('unsupported type (actual: {})'.format(type(a)))
    if not isinstance(b, cupyx.scipy.sparse.csr_matrix):
        raise TypeError('unsupported type (actual: {})'.format(type(b)))
    assert a.has_canonical_format
    assert b.has_canonical_format
    if a.shape[1] != b.shape[0]:
        raise ValueError('mismatched shape')
    if d is not None:
        assert d.ndim == 2
        if not isinstance(d, cupyx.scipy.sparse.csr_matrix):
            raise TypeError('unsupported type (actual: {})'.format(type(d)))
        assert d.has_canonical_format
        if a.shape[0] != d.shape[0] or b.shape[1] != d.shape[1]:
            raise ValueError('mismatched shape')
        if _runtime.is_hip and _driver.get_build_version() < 402:
            raise RuntimeError('d != None is supported since ROCm 4.2.0')
    handle = _device.get_cusparse_handle()
    m, k = a.shape
    _, n = b.shape
    if d is None:
        a, b = _cast_common_type(a, b)
    else:
        a, b, d = _cast_common_type(a, b, d)
    info = _cusparse.createCsrgemm2Info()
    alpha = _numpy.array(alpha, a.dtype).ctypes
    null_ptr = 0
    if d is None:
        beta_data = null_ptr
        d_descr = MatDescriptor.create()
        d_nnz = 0
        d_data = null_ptr
        d_indptr = null_ptr
        d_indices = null_ptr
    else:
        beta = _numpy.array(beta, a.dtype).ctypes
        beta_data = beta.data
        d_descr = d._descr
        d_nnz = d.nnz
        d_data = d.data.data.ptr
        d_indptr = d.indptr.data.ptr
        d_indices = d.indices.data.ptr
    buff_size = _call_cusparse('csrgemm2_bufferSizeExt', a.dtype, handle, m, n, k, alpha.data, a._descr.descriptor, a.nnz, a.indptr.data.ptr, a.indices.data.ptr, b._descr.descriptor, b.nnz, b.indptr.data.ptr, b.indices.data.ptr, beta_data, d_descr.descriptor, d_nnz, d_indptr, d_indices, info)
    buff = _cupy.empty(buff_size, _numpy.int8)
    c_nnz = _numpy.empty((), 'i')
    _cusparse.setPointerMode(handle, _cusparse.CUSPARSE_POINTER_MODE_HOST)
    c_descr = MatDescriptor.create()
    c_indptr = _cupy.empty(m + 1, 'i')
    _cusparse.xcsrgemm2Nnz(handle, m, n, k, a._descr.descriptor, a.nnz, a.indptr.data.ptr, a.indices.data.ptr, b._descr.descriptor, b.nnz, b.indptr.data.ptr, b.indices.data.ptr, d_descr.descriptor, d_nnz, d_indptr, d_indices, c_descr.descriptor, c_indptr.data.ptr, c_nnz.ctypes.data, info, buff.data.ptr)
    c_indices = _cupy.empty(int(c_nnz), 'i')
    c_data = _cupy.empty(int(c_nnz), a.dtype)
    _call_cusparse('csrgemm2', a.dtype, handle, m, n, k, alpha.data, a._descr.descriptor, a.nnz, a.data.data.ptr, a.indptr.data.ptr, a.indices.data.ptr, b._descr.descriptor, b.nnz, b.data.data.ptr, b.indptr.data.ptr, b.indices.data.ptr, beta_data, d_descr.descriptor, d_nnz, d_data, d_indptr, d_indices, c_descr.descriptor, c_data.data.ptr, c_indptr.data.ptr, c_indices.data.ptr, info, buff.data.ptr)
    c = cupyx.scipy.sparse.csr_matrix((c_data, c_indices, c_indptr), shape=(m, n))
    c._has_canonical_format = True
    _cusparse.destroyCsrgemm2Info(info)
    return c