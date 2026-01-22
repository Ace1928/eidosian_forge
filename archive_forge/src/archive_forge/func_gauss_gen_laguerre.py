from sympy.core import S, Dummy, pi
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.gamma_functions import gamma
from sympy.polys.orthopolys import (legendre_poly, laguerre_poly,
from sympy.polys.rootoftools import RootOf
def gauss_gen_laguerre(n, alpha, n_digits):
    """
    Computes the generalized Gauss-Laguerre quadrature [1]_ points and weights.

    Explanation
    ===========

    The generalized Gauss-Laguerre quadrature approximates the integral:

    .. math::
        \\int_{0}^\\infty x^{\\alpha} e^{-x} f(x)\\,dx \\approx
            \\sum_{i=1}^n w_i f(x_i)

    The nodes `x_i` of an order `n` quadrature rule are the roots of
    `L^{\\alpha}_n` and the weights `w_i` are given by:

    .. math::
        w_i = \\frac{\\Gamma(\\alpha+n)}
                {n \\Gamma(n) L^{\\alpha}_{n-1}(x_i) L^{\\alpha+1}_{n-1}(x_i)}

    Parameters
    ==========

    n :
        The order of quadrature.

    alpha :
        The exponent of the singularity, `\\alpha > -1`.

    n_digits :
        Number of significant digits of the points and weights to return.

    Returns
    =======

    (x, w) : the ``x`` and ``w`` are lists of points and weights as Floats.
             The points `x_i` and weights `w_i` are returned as ``(x, w)``
             tuple of lists.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.integrals.quadrature import gauss_gen_laguerre
    >>> x, w = gauss_gen_laguerre(3, -S.Half, 5)
    >>> x
    [0.19016, 1.7845, 5.5253]
    >>> w
    [1.4493, 0.31413, 0.00906]

    >>> x, w = gauss_gen_laguerre(4, 3*S.Half, 5)
    >>> x
    [0.97851, 2.9904, 6.3193, 11.712]
    >>> w
    [0.53087, 0.67721, 0.11895, 0.0023152]

    See Also
    ========

    gauss_legendre, gauss_laguerre, gauss_hermite, gauss_chebyshev_t, gauss_chebyshev_u, gauss_jacobi, gauss_lobatto

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gauss%E2%80%93Laguerre_quadrature
    .. [2] https://people.sc.fsu.edu/~jburkardt/cpp_src/gen_laguerre_rule/gen_laguerre_rule.html
    """
    x = Dummy('x')
    p = laguerre_poly(n, x, alpha=alpha, polys=True)
    p1 = laguerre_poly(n - 1, x, alpha=alpha, polys=True)
    p2 = laguerre_poly(n - 1, x, alpha=alpha + 1, polys=True)
    xi = []
    w = []
    for r in p.real_roots():
        if isinstance(r, RootOf):
            r = r.eval_rational(S.One / 10 ** (n_digits + 2))
        xi.append(r.n(n_digits))
        w.append((gamma(alpha + n) / (n * gamma(n) * p1.subs(x, r) * p2.subs(x, r))).n(n_digits))
    return (xi, w)