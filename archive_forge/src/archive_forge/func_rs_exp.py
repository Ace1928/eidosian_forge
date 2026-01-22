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
def rs_exp(p, x, prec):
    """
    Exponentiation of a series modulo ``O(x**prec)``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_exp
    >>> R, x = ring('x', QQ)
    >>> rs_exp(x**2, x, 7)
    1/6*x**6 + 1/2*x**4 + x**2 + 1
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_exp, p, x, prec)
    R = p.ring
    c = _get_constant_term(p, x)
    if c:
        if R.domain is EX:
            c_expr = c.as_expr()
            const = exp(c_expr)
        elif isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                const = R(exp(c_expr))
            except ValueError:
                R = R.add_gens([exp(c_expr)])
                p = p.set_ring(R)
                x = x.set_ring(R)
                c = c.set_ring(R)
                const = R(exp(c_expr))
        else:
            try:
                const = R(exp(c))
            except ValueError:
                raise DomainError('The given series cannot be expanded in this domain.')
        p1 = p - c
        return const * rs_exp(p1, x, prec)
    if len(p) > 20:
        return _exp1(p, x, prec)
    one = R(1)
    n = 1
    c = []
    for k in range(prec):
        c.append(one / n)
        k += 1
        n *= k
    r = rs_series_from_list(p, c, x, prec)
    return r