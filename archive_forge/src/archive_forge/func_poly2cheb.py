import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def poly2cheb(pol):
    """
    Convert a polynomial to a Chebyshev series.

    Convert an array representing the coefficients of a polynomial (relative
    to the "standard" basis) ordered from lowest degree to highest, to an
    array of the coefficients of the equivalent Chebyshev series, ordered
    from lowest to highest degree.

    Parameters
    ----------
    pol : array_like
        1-D array containing the polynomial coefficients

    Returns
    -------
    c : ndarray
        1-D array containing the coefficients of the equivalent Chebyshev
        series.

    See Also
    --------
    cheb2poly

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy import polynomial as P
    >>> p = P.Polynomial(range(4))
    >>> p
    Polynomial([0., 1., 2., 3.], domain=[-1,  1], window=[-1,  1])
    >>> c = p.convert(kind=P.Chebyshev)
    >>> c
    Chebyshev([1.  , 3.25, 1.  , 0.75], domain=[-1.,  1.], window=[-1.,  1.])
    >>> P.chebyshev.poly2cheb(range(4))
    array([1.  , 3.25, 1.  , 0.75])

    """
    [pol] = pu.as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = chebadd(chebmulx(res), pol[i])
    return res