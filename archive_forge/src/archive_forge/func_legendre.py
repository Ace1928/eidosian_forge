import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def legendre(n, monic=False):
    """Legendre polynomial.

    Defined to be the solution of

    .. math::
        \\frac{d}{dx}\\left[(1 - x^2)\\frac{d}{dx}P_n(x)\\right]
          + n(n + 1)P_n(x) = 0;

    :math:`P_n(x)` is a polynomial of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    P : orthopoly1d
        Legendre polynomial.

    Notes
    -----
    The polynomials :math:`P_n` are orthogonal over :math:`[-1, 1]`
    with weight function 1.

    Examples
    --------
    Generate the 3rd-order Legendre polynomial 1/2*(5x^3 + 0x^2 - 3x + 0):

    >>> from scipy.special import legendre
    >>> legendre(3)
    poly1d([ 2.5,  0. , -1.5,  0. ])

    """
    if n < 0:
        raise ValueError('n must be nonnegative.')
    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    x, w = roots_legendre(n1)
    if n == 0:
        x, w = ([], [])
    hn = 2.0 / (2 * n + 1)
    kn = _gam(2 * n + 1) / _gam(n + 1) ** 2 / 2.0 ** n
    p = orthopoly1d(x, w, hn, kn, wfunc=lambda x: 1.0, limits=(-1, 1), monic=monic, eval_func=lambda x: _ufuncs.eval_legendre(n, x))
    return p