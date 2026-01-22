import numpy
import cupy
from cupy import _core
def hanning(M):
    """Returns the Hanning window.

    The Hanning window is defined as

    .. math::
        w(n) = 0.5 - 0.5\\cos\\left(\\frac{2\\pi{n}}{M-1}\\right)
        \\qquad 0 \\leq n \\leq M-1

    Args:
        M (:class:`~int`):
            Number of points in the output window. If zero or less, an empty
            array is returned.

    Returns:
        ~cupy.ndarray: Output ndarray.

    .. seealso:: :func:`numpy.hanning`
    """
    if M == 1:
        return cupy.ones(1, dtype=cupy.float64)
    if M <= 0:
        return cupy.array([])
    alpha = numpy.pi * 2 / (M - 1)
    out = cupy.empty(M, dtype=cupy.float64)
    return _hanning_kernel(alpha, out)