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
def csrmv(a, x, y=None, alpha=1, beta=0, transa=False):
    """Matrix-vector product for a CSR-matrix and a dense vector.

    .. math::

       y = \\alpha * o_a(A) x + \\beta y,

    where :math:`o_a` is a transpose function when ``transa`` is ``True`` and
    is an identity function otherwise.

    Args:
        a (cupyx.cusparse.csr_matrix): Matrix A.
        x (cupy.ndarray): Vector x.
        y (cupy.ndarray or None): Vector y. It must be F-contiguous.
        alpha (float): Coefficient for x.
        beta (float): Coefficient for y.
        transa (bool): If ``True``, transpose of ``A`` is used.

    Returns:
        cupy.ndarray: Calculated ``y``.

    """
    if not check_availability('csrmv'):
        raise RuntimeError('csrmv is not available.')
    assert y is None or y.flags.f_contiguous
    a_shape = a.shape if not transa else a.shape[::-1]
    if a_shape[1] != len(x):
        raise ValueError('dimension mismatch')
    handle = _device.get_cusparse_handle()
    m, n = a_shape
    a, x, y = _cast_common_type(a, x, y)
    dtype = a.dtype
    if y is None:
        y = _cupy.zeros(m, dtype)
    alpha = _numpy.array(alpha, dtype).ctypes
    beta = _numpy.array(beta, dtype).ctypes
    _call_cusparse('csrmv', dtype, handle, _transpose_flag(transa), a.shape[0], a.shape[1], a.nnz, alpha.data, a._descr.descriptor, a.data.data.ptr, a.indptr.data.ptr, a.indices.data.ptr, x.data.ptr, beta.data, y.data.ptr)
    return y