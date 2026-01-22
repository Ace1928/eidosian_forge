import functools
import warnings
import numpy
import cupy
import cupyx.scipy.fft
@_wraps_polyroutine
def polymul(a1, a2):
    """Computes the product of two polynomials.

    Args:
        a1 (scalar, cupy.ndarray or cupy.poly1d): first input polynomial.
        a2 (scalar, cupy.ndarray or cupy.poly1d): second input polynomial.

    Returns:
        cupy.ndarray or cupy.poly1d: The product of the inputs.

    .. seealso:: :func:`numpy.polymul`

    """
    a1 = cupy.trim_zeros(a1, trim='f')
    a2 = cupy.trim_zeros(a2, trim='f')
    if a1.size == 0:
        a1 = cupy.array([0.0], a1.dtype)
    if a2.size == 0:
        a2 = cupy.array([0.0], a2.dtype)
    return cupy.convolve(a1, a2)