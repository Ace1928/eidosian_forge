import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def lagcompanion(c):
    """
    Return the companion matrix of c.

    The usual companion matrix of the Laguerre polynomials is already
    symmetric when `c` is a basis Laguerre polynomial, so no scaling is
    applied.

    Parameters
    ----------
    c : array_like
        1-D array of Laguerre series coefficients ordered from low to high
        degree.

    Returns
    -------
    mat : ndarray
        Companion matrix of dimensions (deg, deg).

    Notes
    -----

    .. versionadded:: 1.7.0

    """
    [c] = pu.as_series([c])
    if len(c) < 2:
        raise ValueError('Series must have maximum degree of at least 1.')
    if len(c) == 2:
        return np.array([[1 + c[0] / c[1]]])
    n = len(c) - 1
    mat = np.zeros((n, n), dtype=c.dtype)
    top = mat.reshape(-1)[1::n + 1]
    mid = mat.reshape(-1)[0::n + 1]
    bot = mat.reshape(-1)[n::n + 1]
    top[...] = -np.arange(1, n)
    mid[...] = 2.0 * np.arange(n) + 1.0
    bot[...] = top
    mat[:, -1] += c[:-1] / c[-1] * n
    return mat