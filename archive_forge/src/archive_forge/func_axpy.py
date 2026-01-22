import numpy
from numpy import linalg
import warnings
import cupy
from cupy import _core
from cupy_backends.cuda.libs import cublas
from cupy.cuda import device
from cupy.linalg import _util
def axpy(a, x, y):
    """Computes y += a * x.

    (*) y will be updated.
    """
    _check_two_vectors(x, y)
    dtype = x.dtype.char
    if dtype == 'f':
        func = cublas.saxpy
    elif dtype == 'd':
        func = cublas.daxpy
    elif dtype == 'F':
        func = cublas.caxpy
    elif dtype == 'D':
        func = cublas.zaxpy
    else:
        raise TypeError('invalid dtype')
    handle = device.get_cublas_handle()
    a, a_ptr, orig_mode = _setup_scalar_ptr(handle, a, dtype)
    try:
        func(handle, x.size, a_ptr, x.data.ptr, 1, y.data.ptr, 1)
    finally:
        cublas.setPointerMode(handle, orig_mode)