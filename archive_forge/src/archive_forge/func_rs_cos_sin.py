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
def rs_cos_sin(p, x, prec):
    """
    Return the tuple ``(rs_cos(p, x, prec)`, `rs_sin(p, x, prec))``.

    Is faster than calling rs_cos and rs_sin separately
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_cos_sin, p, x, prec)
    t = rs_tan(p / 2, x, prec)
    t2 = rs_square(t, x, prec)
    p1 = rs_series_inversion(1 + t2, x, prec)
    return (rs_mul(p1, 1 - t2, x, prec), rs_mul(p1, 2 * t, x, prec))