import numpy
from numpy import linalg
import warnings
import cupy
from cupy import _core
from cupy_backends.cuda.libs import cublas
from cupy.cuda import device
from cupy.linalg import _util
def nrm2(x, out=None):
    """Computes the Euclidean norm of vector x."""
    if x.ndim != 1:
        raise ValueError('x must be a 1D array (actual: {})'.format(x.ndim))
    dtype = x.dtype.char
    if dtype == 'f':
        func = cublas.snrm2
    elif dtype == 'd':
        func = cublas.dnrm2
    elif dtype == 'F':
        func = cublas.scnrm2
    elif dtype == 'D':
        func = cublas.dznrm2
    else:
        raise TypeError('invalid dtype')
    handle = device.get_cublas_handle()
    result_dtype = dtype.lower()
    result_ptr, result, orig_mode = _setup_result_ptr(handle, out, result_dtype)
    try:
        func(handle, x.size, x.data.ptr, 1, result_ptr)
    finally:
        cublas.setPointerMode(handle, orig_mode)
    if out is None:
        out = result
    elif out.dtype != result_dtype:
        _core.elementwise_copy(result, out)
    return out