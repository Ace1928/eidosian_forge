import numpy
from numpy import linalg
import warnings
import cupy
from cupy import _core
from cupy_backends.cuda.libs import cublas
from cupy.cuda import device
from cupy.linalg import _util
def syrk(trans, a, out=None, alpha=1.0, beta=0.0, lower=False):
    """Computes out := alpha*op1(a)*op2(a) + beta*out

    op1(a) = a if trans is 'N', op2(a) = a.T if transa is 'N'
    op1(a) = a.T if trans is 'T', op2(a) = a if transa is 'T'
    lower specifies  whether  the  upper  or  lower triangular
    part  of the  array  out  is to be  referenced
    """
    assert a.ndim == 2
    dtype = a.dtype.char
    if dtype == 'f':
        func = cublas.ssyrk
    elif dtype == 'd':
        func = cublas.dsyrk
    elif dtype == 'F':
        func = cublas.csyrk
    elif dtype == 'D':
        func = cublas.zsyrk
    else:
        raise TypeError('invalid dtype')
    trans = _trans_to_cublas_op(trans)
    if trans == cublas.CUBLAS_OP_N:
        n, k = a.shape
    else:
        k, n = a.shape
    if out is None:
        out = cupy.zeros((n, n), dtype=dtype, order='F')
        beta = 0.0
    else:
        assert out.ndim == 2
        assert out.shape == (n, n)
        assert out.dtype == dtype
    if lower:
        uplo = cublas.CUBLAS_FILL_MODE_LOWER
    else:
        uplo = cublas.CUBLAS_FILL_MODE_UPPER
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
    lda, trans = _decide_ld_and_trans(a, trans)
    ldo, _ = _decide_ld_and_trans(out, trans)
    if out._c_contiguous:
        if not a._c_contiguous:
            a = a.copy(order='C')
            trans = 1 - trans
            lda = a.shape[1]
        try:
            func(handle, 1 - uplo, trans, n, k, alpha_ptr, a.data.ptr, lda, beta_ptr, out.data.ptr, ldo)
        finally:
            cublas.setPointerMode(handle, orig_mode)
    else:
        if not a._f_contiguous:
            a = a.copy(order='F')
            lda = a.shape[0]
            trans = 1 - trans
        c = out
        if not out._f_contiguous:
            c = out.copy(order='F')
        try:
            func(handle, uplo, trans, n, k, alpha_ptr, a.data.ptr, lda, beta_ptr, out.data.ptr, ldo)
        finally:
            cublas.setPointerMode(handle, orig_mode)
        if not out._f_contiguous:
            out[...] = c
    return out