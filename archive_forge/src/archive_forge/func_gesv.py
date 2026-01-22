import numpy as _numpy
import cupy as _cupy
from cupy_backends.cuda.libs import cublas as _cublas
from cupy_backends.cuda.libs import cusolver as _cusolver
from cupy.cuda import device as _device
import cupyx.cusolver
def gesv(a, b):
    """Solve a linear matrix equation using cusolverDn<t>getr[fs]().

    Computes the solution to a system of linear equation ``ax = b``.

    Args:
        a (cupy.ndarray): The matrix with dimension ``(M, M)``.
        b (cupy.ndarray): The matrix with dimension ``(M)`` or ``(M, K)``.

    Returns:
        cupy.ndarray:
            The matrix with dimension ``(M)`` or ``(M, K)``.

    Note: ``a`` and ``b`` will be overwritten.
    """
    if a.ndim != 2:
        raise ValueError('a.ndim must be 2 (actual: {})'.format(a.ndim))
    if b.ndim not in (1, 2):
        raise ValueError('b.ndim must be 1 or 2 (actual: {})'.format(b.ndim))
    if a.shape[0] != a.shape[1]:
        raise ValueError('a must be a square matrix.')
    if a.shape[0] != b.shape[0]:
        raise ValueError('shape mismatch (a: {}, b: {}).'.format(a.shape, b.shape))
    if a.dtype != b.dtype:
        raise TypeError('dtype mismatch (a: {}, b: {})'.format(a.dtype, b.dtype))
    dtype = a.dtype
    if dtype == 'f':
        t = 's'
    elif dtype == 'd':
        t = 'd'
    elif dtype == 'F':
        t = 'c'
    elif dtype == 'D':
        t = 'z'
    else:
        raise TypeError('unsupported dtype (actual:{})'.format(a.dtype))
    helper = getattr(_cusolver, t + 'getrf_bufferSize')
    getrf = getattr(_cusolver, t + 'getrf')
    getrs = getattr(_cusolver, t + 'getrs')
    n = b.shape[0]
    nrhs = b.shape[1] if b.ndim == 2 else 1
    if a._f_contiguous:
        trans = _cublas.CUBLAS_OP_N
    elif a._c_contiguous:
        trans = _cublas.CUBLAS_OP_T
    else:
        raise ValueError('a must be F-contiguous or C-contiguous.')
    if not b._f_contiguous:
        raise ValueError('b must be F-contiguous.')
    handle = _device.get_cusolver_handle()
    dipiv = _cupy.empty(n, dtype=_numpy.int32)
    dinfo = _cupy.empty(1, dtype=_numpy.int32)
    lwork = helper(handle, n, n, a.data.ptr, n)
    dwork = _cupy.empty(lwork, dtype=a.dtype)
    getrf(handle, n, n, a.data.ptr, n, dwork.data.ptr, dipiv.data.ptr, dinfo.data.ptr)
    _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(getrf, dinfo)
    getrs(handle, trans, n, nrhs, a.data.ptr, n, dipiv.data.ptr, b.data.ptr, n, dinfo.data.ptr)
    _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(getrs, dinfo)