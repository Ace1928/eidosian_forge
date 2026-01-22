import numpy
from numpy import linalg
import warnings
import cupy
from cupy import _core
from cupy_backends.cuda.libs import cublas
from cupy.cuda import device
from cupy.linalg import _util
def gemm(transa, transb, a, b, out=None, alpha=1.0, beta=0.0):
    """Computes out = alpha * op(a) @ op(b) + beta * out

    op(a) = a if transa is 'N', op(a) = a.T if transa is 'T',
    op(a) = a.T.conj() if transa is 'H'.
    op(b) = b if transb is 'N', op(b) = b.T if transb is 'T',
    op(b) = b.T.conj() if transb is 'H'.
    """
    assert a.ndim == b.ndim == 2
    assert a.dtype == b.dtype
    dtype = a.dtype.char
    if dtype == 'f':
        func = cublas.sgemm
    elif dtype == 'd':
        func = cublas.dgemm
    elif dtype == 'F':
        func = cublas.cgemm
    elif dtype == 'D':
        func = cublas.zgemm
    else:
        raise TypeError('invalid dtype')
    transa = _trans_to_cublas_op(transa)
    transb = _trans_to_cublas_op(transb)
    if transa == cublas.CUBLAS_OP_N:
        m, k = a.shape
    else:
        k, m = a.shape
    if transb == cublas.CUBLAS_OP_N:
        n = b.shape[1]
        assert b.shape[0] == k
    else:
        n = b.shape[0]
        assert b.shape[1] == k
    if out is None:
        out = cupy.empty((m, n), dtype=dtype, order='F')
        beta = 0.0
    else:
        assert out.ndim == 2
        assert out.shape == (m, n)
        assert out.dtype == dtype
    alpha, alpha_ptr = _get_scalar_ptr(alpha, a.dtype)
    beta, beta_ptr = _get_scalar_ptr(beta, a.dtype)
    handle = device.get_cublas_handle()
    orig_mode = cublas.getPointerMode(handle)
    if isinstance(alpha, cupy.ndarray) or isinstance(beta, cupy.ndarray):
        if not isinstance(alpha, cupy.ndarray):
            alpha = cupy.array(alpha)
            alpha_ptr = alpha.data.ptr
        if not isinstance(beta, cupy.ndarray):
            beta = cupy.array(beta)
            beta_ptr = beta.data.ptr
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)
    else:
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    lda, transa = _decide_ld_and_trans(a, transa)
    ldb, transb = _decide_ld_and_trans(b, transb)
    if not (lda is None or ldb is None):
        if out._f_contiguous:
            try:
                func(handle, transa, transb, m, n, k, alpha_ptr, a.data.ptr, lda, b.data.ptr, ldb, beta_ptr, out.data.ptr, m)
            finally:
                cublas.setPointerMode(handle, orig_mode)
            return out
        elif out._c_contiguous:
            try:
                func(handle, 1 - transb, 1 - transa, n, m, k, alpha_ptr, b.data.ptr, ldb, a.data.ptr, lda, beta_ptr, out.data.ptr, n)
            finally:
                cublas.setPointerMode(handle, orig_mode)
            return out
    a, lda = _change_order_if_necessary(a, lda)
    b, ldb = _change_order_if_necessary(b, ldb)
    c = out
    if not out._f_contiguous:
        c = out.copy(order='F')
    try:
        func(handle, transa, transb, m, n, k, alpha_ptr, a.data.ptr, lda, b.data.ptr, ldb, beta_ptr, c.data.ptr, m)
    finally:
        cublas.setPointerMode(handle, orig_mode)
    if not out._f_contiguous:
        _core.elementwise_copy(c, out)
    return out