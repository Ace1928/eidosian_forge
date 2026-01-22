import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def polymulx(c):
    """Multiply a polynomial by x.

    Multiply the polynomial `c` by x, where x is the independent
    variable.


    Parameters
    ----------
    c : array_like
        1-D array of polynomial coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the result of the multiplication.

    See Also
    --------
    polyadd, polysub, polymul, polydiv, polypow

    Notes
    -----

    .. versionadded:: 1.5.0

    """
    [c] = pu.as_series([c])
    if len(c) == 1 and c[0] == 0:
        return c
    prd = np.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0] * 0
    prd[1:] = c
    return prd