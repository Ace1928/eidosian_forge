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
def rs_series_inversion(p, x, prec):
    """
    Multivariate series inversion ``1/p`` modulo ``O(x**prec)``.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_series_inversion
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_series_inversion(1 + x*y**2, x, 4)
    -x**3*y**6 + x**2*y**4 - x*y**2 + 1
    >>> rs_series_inversion(1 + x*y**2, y, 4)
    -x*y**2 + 1
    >>> rs_series_inversion(x + x**2, x, 4)
    x**3 - x**2 + x - 1 + x**(-1)
    """
    R = p.ring
    if p == R.zero:
        raise ZeroDivisionError
    zm = R.zero_monom
    index = R.gens.index(x)
    m = min(p, key=lambda k: k[index])[index]
    if m:
        p = mul_xin(p, index, -m)
        prec = prec + m
    if zm not in p:
        raise NotImplementedError('No constant term in series')
    if _has_constant_term(p - p[zm], x):
        raise NotImplementedError('p - p[0] must not have a constant term in the series variables')
    r = _series_inversion1(p, x, prec)
    if m != 0:
        r = mul_xin(r, index, -m)
    return r