from sympy.core import S, Dummy, pi
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.gamma_functions import gamma
from sympy.polys.orthopolys import (legendre_poly, laguerre_poly,
from sympy.polys.rootoftools import RootOf
def gauss_chebyshev_u(n, n_digits):
    """
    Computes the Gauss-Chebyshev quadrature [1]_ points and weights of
    the second kind.

    Explanation
    ===========

    The Gauss-Chebyshev quadrature of the second kind approximates the
    integral:

    .. math::
        \\int_{-1}^{1} \\sqrt{1-x^2} f(x)\\,dx \\approx \\sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of `U_n`
    and the weights `w_i` are given by:

    .. math::
        w_i = \\frac{\\pi}{n+1} \\sin^2 \\left(\\frac{i}{n+1}\\pi\\right)

    Parameters
    ==========

    n : the order of quadrature

    n_digits : number of significant digits of the points and weights to return

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy.integrals.quadrature import gauss_chebyshev_u
    >>> x, w = gauss_chebyshev_u(3, 5)
    >>> x
    [0.70711, 0, -0.70711]
    >>> w
    [0.3927, 0.7854, 0.3927]

    >>> x, w = gauss_chebyshev_u(6, 5)
    >>> x
    [0.90097, 0.62349, 0.22252, -0.22252, -0.62349, -0.90097]
    >>> w
    [0.084489, 0.27433, 0.42658, 0.42658, 0.27433, 0.084489]

    See Also
    ========

    gauss_legendre, gauss_laguerre, gauss_hermite, gauss_gen_laguerre, gauss_chebyshev_t, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Chebyshev%E2%80%93Gauss_quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/chebyshev2_rule/chebyshev2_rule.html
    """
    xi = []
    w = []
    for i in range(1, n + 1):
        xi.append(cos(i / (n + S.One) * S.Pi).n(n_digits))
        w.append((S.Pi / (n + S.One) * sin(i * S.Pi / (n + S.One)) ** 2).n(n_digits))
    return (xi, w)