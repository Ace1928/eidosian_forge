from sympy.core.symbol import Dummy
from sympy.polys.densearith import (dup_mul, dup_mul_ground,
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polytools import named_poly
from sympy.utilities import public
def dup_spherical_bessel_fn_minus(n, K):
    """Low-level implementation of fn(-n, x)."""
    m2, m1 = ([K.one, K.zero], [K.zero])
    for i in range(2, n + 1):
        m2, m1 = (m1, dup_sub(dup_mul_ground(dup_lshift(m1, 1, K), K(3 - 2 * i), K), m2, K))
    return m1