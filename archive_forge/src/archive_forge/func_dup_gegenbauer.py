from sympy.core.symbol import Dummy
from sympy.polys.densearith import (dup_mul, dup_mul_ground,
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polytools import named_poly
from sympy.utilities import public
def dup_gegenbauer(n, a, K):
    """Low-level implementation of Gegenbauer polynomials."""
    if n < 1:
        return [K.one]
    m2, m1 = ([K.one], [K(2) * a, K.zero])
    for i in range(2, n + 1):
        p1 = dup_mul_ground(dup_lshift(m1, 1, K), K(2) * (a - K.one) / K(i) + K(2), K)
        p2 = dup_mul_ground(m2, K(2) * (a - K.one) / K(i) + K.one, K)
        m2, m1 = (m1, dup_sub(p1, p2, K))
    return m1