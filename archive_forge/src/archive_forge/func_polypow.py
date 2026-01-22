import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def polypow(c, pow, maxpower=None):
    """Raise a polynomial to a power.

    Returns the polynomial `c` raised to the power `pow`. The argument
    `c` is a sequence of coefficients ordered from low to high. i.e.,
    [1,2,3] is the series  ``1 + 2*x + 3*x**2.``

    Parameters
    ----------
    c : array_like
        1-D array of array of series coefficients ordered from low to
        high degree.
    pow : integer
        Power to which the series will be raised
    maxpower : integer, optional
        Maximum power allowed. This is mainly to limit growth of the series
        to unmanageable size. Default is 16

    Returns
    -------
    coef : ndarray
        Power series of power.

    See Also
    --------
    polyadd, polysub, polymulx, polymul, polydiv

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> P.polypow([1,2,3], 2)
    array([ 1., 4., 10., 12., 9.])

    """
    return pu._pow(np.convolve, c, pow, maxpower)