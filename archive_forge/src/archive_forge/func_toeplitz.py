import math
import cupy
from cupy import _core
from cupyx.scipy.linalg import _uarray
@_uarray.implements('toeplitz')
def toeplitz(c, r=None):
    """Construct a Toeplitz matrix.

    The Toeplitz matrix has constant diagonals, with ``c`` as its first column
    and ``r`` as its first row. If ``r`` is not given, ``r == conjugate(c)`` is
    assumed.

    Args:
        c (cupy.ndarray): First column of the matrix. Whatever the actual shape
            of ``c``, it will be converted to a 1-D array.
        r (cupy.ndarray, optional): First row of the matrix. If None,
            ``r = conjugate(c)`` is assumed; in this case, if ``c[0]`` is real,
            the result is a Hermitian matrix. r[0] is ignored; the first row of
            the returned matrix is ``[c[0], r[1:]]``. Whatever the actual shape
            of ``r``, it will be converted to a 1-D array.

    Returns:
        cupy.ndarray: The Toeplitz matrix. Dtype is the same as
        ``(c[0] + r[0]).dtype``.

    .. seealso:: :func:`cupyx.scipy.linalg.circulant`
    .. seealso:: :func:`cupyx.scipy.linalg.hankel`
    .. seealso:: :func:`cupyx.scipy.linalg.solve_toeplitz`
    .. seealso:: :func:`cupyx.scipy.linalg.fiedler`
    .. seealso:: :func:`scipy.linalg.toeplitz`
    """
    c = c.ravel()
    r = c.conjugate() if r is None else r.ravel()
    return _create_toeplitz_matrix(c[::-1], r[1:])