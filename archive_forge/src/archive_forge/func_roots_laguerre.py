import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def roots_laguerre(n, mu=False):
    """Gauss-Laguerre quadrature.

    Compute the sample points and weights for Gauss-Laguerre
    quadrature. The sample points are the roots of the nth degree
    Laguerre polynomial, :math:`L_n(x)`. These sample points and
    weights correctly integrate polynomials of degree :math:`2n - 1`
    or less over the interval :math:`[0, \\infty]` with weight function
    :math:`w(x) = e^{-x}`. See 22.2.13 in [AS]_ for details.

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
    numpy.polynomial.laguerre.laggauss

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    return roots_genlaguerre(n, 0.0, mu=mu)