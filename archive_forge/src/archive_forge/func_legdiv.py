import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def legdiv(c1, c2):
    """
    Divide one Legendre series by another.

    Returns the quotient-with-remainder of two Legendre series
    `c1` / `c2`.  The arguments are sequences of coefficients from lowest
    order "term" to highest, e.g., [1,2,3] represents the series
    ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Legendre series coefficients ordered from low to
        high.

    Returns
    -------
    quo, rem : ndarrays
        Of Legendre series coefficients representing the quotient and
        remainder.

    See Also
    --------
    legadd, legsub, legmulx, legmul, legpow

    Notes
    -----
    In general, the (polynomial) division of one Legendre series by another
    results in quotient and remainder terms that are not in the Legendre
    polynomial basis set.  Thus, to express these results as a Legendre
    series, it is necessary to "reproject" the results onto the Legendre
    basis set, which may produce "unintuitive" (but correct) results; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial import legendre as L
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    >>> L.legdiv(c1,c2) # quotient "intuitive," remainder not
    (array([3.]), array([-8., -4.]))
    >>> c2 = (0,1,2,3)
    >>> L.legdiv(c2,c1) # neither "intuitive"
    (array([-0.07407407,  1.66666667]), array([-1.03703704, -2.51851852])) # may vary

    """
    return pu._div(legmul, c1, c2)