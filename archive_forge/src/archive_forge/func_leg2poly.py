import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def leg2poly(c):
    """
    Convert a Legendre series to a polynomial.

    Convert an array representing the coefficients of a Legendre series,
    ordered from lowest degree to highest, to an array of the coefficients
    of the equivalent polynomial (relative to the "standard" basis) ordered
    from lowest to highest degree.

    Parameters
    ----------
    c : array_like
        1-D array containing the Legendre series coefficients, ordered
        from lowest order term to highest.

    Returns
    -------
    pol : ndarray
        1-D array containing the coefficients of the equivalent polynomial
        (relative to the "standard" basis) ordered from lowest order term
        to highest.

    See Also
    --------
    poly2leg

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy import polynomial as P
    >>> c = P.Legendre(range(4))
    >>> c
    Legendre([0., 1., 2., 3.], domain=[-1,  1], window=[-1,  1])
    >>> p = c.convert(kind=P.Polynomial)
    >>> p
    Polynomial([-1. , -3.5,  3. ,  7.5], domain=[-1.,  1.], window=[-1.,  1.])
    >>> P.legendre.leg2poly(range(4))
    array([-1. , -3.5,  3. ,  7.5])


    """
    from .polynomial import polyadd, polysub, polymulx
    [c] = pu.as_series([c])
    n = len(c)
    if n < 3:
        return c
    else:
        c0 = c[-2]
        c1 = c[-1]
        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(c[i - 2], c1 * (i - 1) / i)
            c1 = polyadd(tmp, polymulx(c1) * (2 * i - 1) / i)
        return polyadd(c0, polymulx(c1))