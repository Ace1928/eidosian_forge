import math
import warnings
import numpy as np
from numpy.lib.stride_tricks import as_strided
def invhilbert(n, exact=False):
    """
    Compute the inverse of the Hilbert matrix of order `n`.

    The entries in the inverse of a Hilbert matrix are integers. When `n`
    is greater than 14, some entries in the inverse exceed the upper limit
    of 64 bit integers. The `exact` argument provides two options for
    dealing with these large integers.

    Parameters
    ----------
    n : int
        The order of the Hilbert matrix.
    exact : bool, optional
        If False, the data type of the array that is returned is np.float64,
        and the array is an approximation of the inverse.
        If True, the array is the exact integer inverse array. To represent
        the exact inverse when n > 14, the returned array is an object array
        of long integers. For n <= 14, the exact inverse is returned as an
        array with data type np.int64.

    Returns
    -------
    invh : (n, n) ndarray
        The data type of the array is np.float64 if `exact` is False.
        If `exact` is True, the data type is either np.int64 (for n <= 14)
        or object (for n > 14). In the latter case, the objects in the
        array will be long integers.

    See Also
    --------
    hilbert : Create a Hilbert matrix.

    Notes
    -----
    .. versionadded:: 0.10.0

    Examples
    --------
    >>> from scipy.linalg import invhilbert
    >>> invhilbert(4)
    array([[   16.,  -120.,   240.,  -140.],
           [ -120.,  1200., -2700.,  1680.],
           [  240., -2700.,  6480., -4200.],
           [ -140.,  1680., -4200.,  2800.]])
    >>> invhilbert(4, exact=True)
    array([[   16,  -120,   240,  -140],
           [ -120,  1200, -2700,  1680],
           [  240, -2700,  6480, -4200],
           [ -140,  1680, -4200,  2800]], dtype=int64)
    >>> invhilbert(16)[7,7]
    4.2475099528537506e+19
    >>> invhilbert(16, exact=True)[7,7]
    42475099528537378560

    """
    from scipy.special import comb
    if exact:
        if n > 14:
            dtype = object
        else:
            dtype = np.int64
    else:
        dtype = np.float64
    invh = np.empty((n, n), dtype=dtype)
    for i in range(n):
        for j in range(0, i + 1):
            s = i + j
            invh[i, j] = (-1) ** s * (s + 1) * comb(n + i, n - j - 1, exact=exact) * comb(n + j, n - i - 1, exact=exact) * comb(s, i, exact=exact) ** 2
            if i != j:
                invh[j, i] = invh[i, j]
    return invh