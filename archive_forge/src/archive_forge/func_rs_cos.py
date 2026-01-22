from sympy.polys.domains import QQ, EX
from sympy.polys.rings import PolyElement, ring, sring
from sympy.polys.polyerrors import DomainError
from sympy.polys.monomials import (monomial_min, monomial_mul, monomial_div,
from mpmath.libmp.libintmath import ifac
from sympy.core import PoleError, Function, Expr
from sympy.core.numbers import Rational, igcd
from sympy.functions import sin, cos, tan, atan, exp, atanh, tanh, log, ceiling
from sympy.utilities.misc import as_int
from mpmath.libmp.libintmath import giant_steps
import math
def rs_cos(p, x, prec):
    """
    Cosine of a series

    Return the series expansion of the cos of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_cos
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_cos(x + x*y, x, 4)
    -1/2*x**2*y**2 - x**2*y - 1/2*x**2 + 1
    >>> rs_cos(x + x*y, x, 4)/x**QQ(7, 5)
    -1/2*x**(3/5)*y**2 - x**(3/5)*y - 1/2*x**(3/5) + x**(-7/5)

    See Also
    ========

    cos
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_cos, p, x, prec)
    R = p.ring
    c = _get_constant_term(p, x)
    if c:
        if R.domain is EX:
            c_expr = c.as_expr()
            _, _ = (sin(c_expr), cos(c_expr))
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                _, _ = (R(sin(c_expr)), R(cos(c_expr)))
            except ValueError:
                R = R.add_gens([sin(c_expr), cos(c_expr)])
                p = p.set_ring(R)
                x = x.set_ring(R)
                c = c.set_ring(R)
        else:
            try:
                _, _ = (R(sin(c)), R(cos(c)))
            except ValueError:
                raise DomainError('The given series cannot be expanded in this domain.')
        p1 = p - c
        p_cos = rs_cos(p1, x, prec)
        p_sin = rs_sin(p1, x, prec)
        R = R.compose(p_cos.ring).compose(p_sin.ring)
        p_cos.set_ring(R)
        p_sin.set_ring(R)
        t1, t2 = (R(sin(c_expr)), R(cos(c_expr)))
        return p_cos * t2 - p_sin * t1
    if len(p) > 20 and R.ngens == 1:
        t = rs_tan(p / 2, x, prec)
        t2 = rs_square(t, x, prec)
        p1 = rs_series_inversion(1 + t2, x, prec)
        return rs_mul(p1, 1 - t2, x, prec)
    one = R(1)
    n = 1
    c = []
    for k in range(2, prec + 2, 2):
        c.append(one / n)
        c.append(0)
        n *= -k * (k - 1)
    return rs_series_from_list(p, c, x, prec)