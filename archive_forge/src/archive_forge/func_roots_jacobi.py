import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def roots_jacobi(n, alpha, beta, mu=False):
    """Gauss-Jacobi quadrature.

    Compute the sample points and weights for Gauss-Jacobi
    quadrature. The sample points are the roots of the nth degree
    Jacobi polynomial, :math:`P^{\\alpha, \\beta}_n(x)`. These sample
    points and weights correctly integrate polynomials of degree
    :math:`2n - 1` or less over the interval :math:`[-1, 1]` with
    weight function :math:`w(x) = (1 - x)^{\\alpha} (1 +
    x)^{\\beta}`. See 22.2.1 in [AS]_ for details.

    Parameters
    ----------
    n : int
        quadrature order
    alpha : float
        alpha must be > -1
    beta : float
        beta must be > -1
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
    if alpha <= -1 or beta <= -1:
        raise ValueError('alpha and beta must be greater than -1.')
    if alpha == 0.0 and beta == 0.0:
        return roots_legendre(m, mu)
    if alpha == beta:
        return roots_gegenbauer(m, alpha + 0.5, mu)
    if alpha + beta <= 1000:
        mu0 = 2.0 ** (alpha + beta + 1) * _ufuncs.beta(alpha + 1, beta + 1)
    else:
        mu0 = np.exp((alpha + beta + 1) * np.log(2.0) + _ufuncs.betaln(alpha + 1, beta + 1))
    a = alpha
    b = beta
    if a + b == 0.0:

        def an_func(k):
            return np.where(k == 0, (b - a) / (2 + a + b), 0.0)
    else:

        def an_func(k):
            return np.where(k == 0, (b - a) / (2 + a + b), (b * b - a * a) / ((2.0 * k + a + b) * (2.0 * k + a + b + 2)))

    def bn_func(k):
        return 2.0 / (2.0 * k + a + b) * np.sqrt((k + a) * (k + b) / (2 * k + a + b + 1)) * np.where(k == 1, 1.0, np.sqrt(k * (k + a + b) / (2.0 * k + a + b - 1)))

    def f(n, x):
        return _ufuncs.eval_jacobi(n, a, b, x)

    def df(n, x):
        return 0.5 * (n + a + b + 1) * _ufuncs.eval_jacobi(n - 1, a + 1, b + 1, x)
    return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, False, mu)