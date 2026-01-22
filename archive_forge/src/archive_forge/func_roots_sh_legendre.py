import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def roots_sh_legendre(n, mu=False):
    """Gauss-Legendre (shifted) quadrature.

    Compute the sample points and weights for Gauss-Legendre
    quadrature. The sample points are the roots of the nth degree
    shifted Legendre polynomial :math:`P^*_n(x)`. These sample points
    and weights correctly integrate polynomials of degree :math:`2n -
    1` or less over the interval :math:`[0, 1]` with weight function
    :math:`w(x) = 1.0`. See 2.2.11 in [AS]_ for details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.quadrature
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    x, w = roots_legendre(n)
    x = (x + 1) / 2
    w /= 2
    if mu:
        return (x, w, 1.0)
    else:
        return (x, w)