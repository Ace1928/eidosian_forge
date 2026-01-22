import math
import cupy
from cupy import _core
from cupyx.scipy.linalg import _uarray
@_uarray.implements('hankel')
def hankel(c, r=None):
    """Construct a Hankel matrix.

    The Hankel matrix has constant anti-diagonals, with ``c`` as its first
    column and ``r`` as its last row. If ``r`` is not given, then
    ``r = zeros_like(c)`` is assumed.

    Args:
        c (cupy.ndarray): First column of the matrix. Whatever the actual shape
            of ``c``, it will be converted to a 1-D array.
        r (cupy.ndarray, optionnal): Last row of the matrix. If None,
            ``r = zeros_like(c)`` is assumed. ``r[0]`` is ignored; the last row
            of the returned matrix is ``[c[-1], r[1:]]``. Whatever the actual
            shape of ``r``, it will be converted to a 1-D array.

    Returns:
        cupy.ndarray: The Hankel matrix. Dtype is the same as
        ``(c[0] + r[0]).dtype``.

    .. seealso:: :func:`cupyx.scipy.linalg.toeplitz`
    .. seealso:: :func:`cupyx.scipy.linalg.circulant`
    .. seealso:: :func:`scipy.linalg.hankel`
    """
    c = c.ravel()
    r = cupy.zeros_like(c) if r is None else r.ravel()
    return _create_toeplitz_matrix(c, r[1:], True)