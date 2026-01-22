import cupy
import operator
import warnings
Removes small trailing coefficients from a polynomial.

    Args:
        c(cupy.ndarray): 1d array of coefficients from lowest to highest order.
        tol(number, optional): trailing coefficients whose absolute value are
            less than or equal to ``tol`` are trimmed.

    Returns:
        cupy.ndarray: trimmed 1d array.

    .. seealso:: :func:`numpy.polynomial.polyutils.trimcoef`

    