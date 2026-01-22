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
def rs_subs(p, rules, x, prec):
    """
    Substitution with truncation according to the mapping in ``rules``.

    Return a series with precision ``prec`` in the generator ``x``

    Note that substitutions are not done one after the other

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_subs
    >>> R, x, y = ring('x, y', QQ)
    >>> p = x**2 + y**2
    >>> rs_subs(p, {x: x+ y, y: x+ 2*y}, x, 3)
    2*x**2 + 6*x*y + 5*y**2
    >>> (x + y)**2 + (x + 2*y)**2
    2*x**2 + 6*x*y + 5*y**2

    which differs from

    >>> rs_subs(rs_subs(p, {x: x+ y}, x, 3), {y: x+ 2*y}, x, 3)
    5*x**2 + 12*x*y + 8*y**2

    Parameters
    ----------
    p : :class:`~.PolyElement` Input series.
    rules : ``dict`` with substitution mappings.
    x : :class:`~.PolyElement` in which the series truncation is to be done.
    prec : :class:`~.Integer` order of the series after truncation.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_subs
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_subs(x**2+y**2, {y: (x+y)**2}, x, 3)
     6*x**2*y**2 + x**2 + 4*x*y**3 + y**4
    """
    R = p.ring
    ngens = R.ngens
    d = R(0)
    for i in range(ngens):
        d[i, 1] = R.gens[i]
    for var in rules:
        d[R.index(var), 1] = rules[var]
    p1 = R(0)
    p_keys = sorted(p.keys())
    for expv in p_keys:
        p2 = R(1)
        for i in range(ngens):
            power = expv[i]
            if power == 0:
                continue
            if (i, power) not in d:
                q, r = divmod(power, 2)
                if r == 0 and (i, q) in d:
                    d[i, power] = rs_square(d[i, q], x, prec)
                elif (i, power - 1) in d:
                    d[i, power] = rs_mul(d[i, power - 1], d[i, 1], x, prec)
                else:
                    d[i, power] = rs_pow(d[i, 1], power, x, prec)
            p2 = rs_mul(p2, d[i, power], x, prec)
        p1 += p2 * p[expv]
    return p1