import numpy
from numpy import linalg
import warnings
import cupy
from cupy import _core
from cupy_backends.cuda.libs import cublas
from cupy.cuda import device
from cupy.linalg import _util
def dgmm(side, a, x, out=None, incx=1):
    """Computes diag(x) @ a or a @ diag(x)

    Computes diag(x) @ a if side is 'L', a @ diag(x) if side is 'R'.
    """
    assert a.ndim == 2
    assert 0 <= x.ndim <= 2
    assert a.dtype == x.dtype
    dtype = a.dtype.char
    if dtype == 'f':
        func = cublas.sdgmm
    elif dtype == 'd':
        func = cublas.ddgmm
    elif dtype == 'F':
        func = cublas.cdgmm
    elif dtype == 'D':
        func = cublas.zdgmm
    else:
        raise TypeError('invalid dtype')
    if side == 'L' or side == cublas.CUBLAS_SIDE_LEFT:
        side = cublas.CUBLAS_SIDE_LEFT
    elif side == 'R' or side == cublas.CUBLAS_SIDE_RIGHT:
        side = cublas.CUBLAS_SIDE_RIGHT
    else:
        raise ValueError('invalid side (actual: {})'.format(side))
    m, n = a.shape
    if side == cublas.CUBLAS_SIDE_LEFT:
        assert x.size >= (m - 1) * abs(incx) + 1
    else:
        assert x.size >= (n - 1) * abs(incx) + 1
    if out is None:
        if a._c_contiguous:
            order = 'C'
        else:
            order = 'F'
        out = cupy.empty((m, n), dtype=dtype, order=order)
    else:
        assert out.ndim == 2
        assert out.shape == a.shape
        assert out.dtype == a.dtype
    handle = device.get_cublas_handle()
    if out._c_contiguous:
        if not a._c_contiguous:
            a = a.copy(order='C')
        func(handle, 1 - side, n, m, a.data.ptr, n, x.data.ptr, incx, out.data.ptr, n)
    else:
        if not a._f_contiguous:
            a = a.copy(order='F')
        c = out
        if not out._f_contiguous:
            c = out.copy(order='F')
        func(handle, side, m, n, a.data.ptr, m, x.data.ptr, incx, c.data.ptr, m)
        if not out._f_contiguous:
            _core.elementwise_copy(c, out)
    return out