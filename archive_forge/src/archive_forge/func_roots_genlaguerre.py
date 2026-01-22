import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def roots_genlaguerre(n, alpha, mu=False):
    """Gauss-generalized Laguerre quadrature.

    Compute the sample points and weights for Gauss-generalized
    Laguerre quadrature. The sample points are the roots of the nth
    degree generalized Laguerre polynomial, :math:`L^{\\alpha}_n(x)`.
    These sample points and weights correctly integrate polynomials of
    degree :math:`2n - 1` or less over the interval :math:`[0,
    \\infty]` with weight function :math:`w(x) = x^{\\alpha}
    e^{-x}`. See 22.3.9 in [AS]_ for details.

    Parameters
    ----------
    n : int
        quadrature order
    alpha : float
        alpha must be > -1
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
    m = int(n)
    if n < 1 or n != m:
        raise ValueError('n must be a positive integer.')
    if alpha < -1:
        raise ValueError('alpha must be greater than -1.')
    mu0 = _ufuncs.gamma(alpha + 1)
    if m == 1:
        x = np.array([alpha + 1.0], 'd')
        w = np.array([mu0], 'd')
        if mu:
            return (x, w, mu0)
        else:
            return (x, w)

    def an_func(k):
        return 2 * k + alpha + 1

    def bn_func(k):
        return -np.sqrt(k * (k + alpha))

    def f(n, x):
        return _ufuncs.eval_genlaguerre(n, alpha, x)

    def df(n, x):
        return (n * _ufuncs.eval_genlaguerre(n, alpha, x) - (n + alpha) * _ufuncs.eval_genlaguerre(n - 1, alpha, x)) / x
    return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, False, mu)