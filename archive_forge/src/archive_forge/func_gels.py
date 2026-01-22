import numpy as _numpy
import cupy as _cupy
from cupy_backends.cuda.libs import cublas as _cublas
from cupy_backends.cuda.libs import cusolver as _cusolver
from cupy.cuda import device as _device
import cupyx.cusolver
def gels(a, b):
    """Solves over/well/under-determined linear systems.

    Computes least-square solution to equation ``ax = b` by QR factorization
    using cusolverDn<t>geqrf().

    Args:
        a (cupy.ndarray): The matrix with dimension ``(M, N)``.
        b (cupy.ndarray): The matrix with dimension ``(M)`` or ``(M, K)``.

    Returns:
        cupy.ndarray:
            The matrix with dimension ``(N)`` or ``(N, K)``.
    """
    if a.ndim != 2:
        raise ValueError('a.ndim must be 2 (actual: {})'.format(a.ndim))
    if b.ndim == 1:
        nrhs = 1
    elif b.ndim == 2:
        nrhs = b.shape[1]
    else:
        raise ValueError('b.ndim must be 1 or 2 (actual: {})'.format(b.ndim))
    if a.shape[0] != b.shape[0]:
        raise ValueError('shape mismatch (a: {}, b: {}).'.format(a.shape, b.shape))
    if a.dtype != b.dtype:
        raise ValueError('dtype mismatch (a: {}, b: {}).'.format(a.dtype, b.dtype))
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
        raise ValueError('unsupported dtype (actual: {})'.format(dtype))
    geqrf_helper = getattr(_cusolver, t + 'geqrf_bufferSize')
    geqrf = getattr(_cusolver, t + 'geqrf')
    trsm = getattr(_cublas, t + 'trsm')
    if t in 'sd':
        ormqr_helper = getattr(_cusolver, t + 'ormqr_bufferSize')
        ormqr = getattr(_cusolver, t + 'ormqr')
    else:
        ormqr_helper = getattr(_cusolver, t + 'unmqr_bufferSize')
        ormqr = getattr(_cusolver, t + 'unmqr')
    no_trans = _cublas.CUBLAS_OP_N
    if dtype.char in 'fd':
        trans = _cublas.CUBLAS_OP_T
    else:
        trans = _cublas.CUBLAS_OP_C
    m, n = a.shape
    mn_min = min(m, n)
    dev_info = _cupy.empty(1, dtype=_numpy.int32)
    tau = _cupy.empty(mn_min, dtype=dtype)
    cusolver_handle = _device.get_cusolver_handle()
    cublas_handle = _device.get_cublas_handle()
    one = _numpy.array(1.0, dtype=dtype)
    if m >= n:
        a = a.copy(order='F')
        b = b.copy(order='F')
        ws_size = geqrf_helper(cusolver_handle, m, n, a.data.ptr, m)
        workspace = _cupy.empty(ws_size, dtype=dtype)
        geqrf(cusolver_handle, m, n, a.data.ptr, m, tau.data.ptr, workspace.data.ptr, ws_size, dev_info.data.ptr)
        _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(geqrf, dev_info)
        ws_size = ormqr_helper(cusolver_handle, _cublas.CUBLAS_SIDE_LEFT, trans, m, nrhs, mn_min, a.data.ptr, m, tau.data.ptr, b.data.ptr, m)
        workspace = _cupy.empty(ws_size, dtype=dtype)
        ormqr(cusolver_handle, _cublas.CUBLAS_SIDE_LEFT, trans, m, nrhs, mn_min, a.data.ptr, m, tau.data.ptr, b.data.ptr, m, workspace.data.ptr, ws_size, dev_info.data.ptr)
        _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(ormqr, dev_info)
        trsm(cublas_handle, _cublas.CUBLAS_SIDE_LEFT, _cublas.CUBLAS_FILL_MODE_UPPER, no_trans, _cublas.CUBLAS_DIAG_NON_UNIT, mn_min, nrhs, one.ctypes.data, a.data.ptr, m, b.data.ptr, m)
        return b[:n]
    else:
        a = a.conj().T.copy(order='F')
        bb = b
        out_shape = (n,) if b.ndim == 1 else (n, nrhs)
        b = _cupy.zeros(out_shape, dtype=dtype, order='F')
        b[:m] = bb
        ws_size = geqrf_helper(cusolver_handle, n, m, a.data.ptr, n)
        workspace = _cupy.empty(ws_size, dtype=dtype)
        geqrf(cusolver_handle, n, m, a.data.ptr, n, tau.data.ptr, workspace.data.ptr, ws_size, dev_info.data.ptr)
        _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(geqrf, dev_info)
        trsm(cublas_handle, _cublas.CUBLAS_SIDE_LEFT, _cublas.CUBLAS_FILL_MODE_UPPER, trans, _cublas.CUBLAS_DIAG_NON_UNIT, m, nrhs, one.ctypes.data, a.data.ptr, n, b.data.ptr, n)
        ws_size = ormqr_helper(cusolver_handle, _cublas.CUBLAS_SIDE_LEFT, no_trans, n, nrhs, mn_min, a.data.ptr, n, tau.data.ptr, b.data.ptr, n)
        workspace = _cupy.empty(ws_size, dtype=dtype)
        ormqr(cusolver_handle, _cublas.CUBLAS_SIDE_LEFT, no_trans, n, nrhs, mn_min, a.data.ptr, n, tau.data.ptr, b.data.ptr, n, workspace.data.ptr, ws_size, dev_info.data.ptr)
        _cupy.linalg._util._check_cusolver_dev_info_if_synchronization_allowed(ormqr, dev_info)
        return b