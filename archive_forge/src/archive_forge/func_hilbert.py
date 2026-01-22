import math
import cupy
from cupy import _core
from cupyx.scipy.linalg import _uarray
@_uarray.implements('hilbert')
def hilbert(n):
    """Create a Hilbert matrix of order ``n``.

    Returns the ``n`` by ``n`` array with entries ``h[i,j] = 1 / (i + j + 1)``.

    Args:
        n (int): The size of the array to create.

    Returns:
        cupy.ndarray: The Hilbert matrix.

    .. seealso:: :func:`scipy.linalg.hilbert`
    """
    values = cupy.arange(1, 2 * n, dtype=cupy.float64)
    cupy.reciprocal(values, values)
    return hankel(values[:n], r=values[n - 1:])