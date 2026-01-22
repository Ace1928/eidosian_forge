from sympy.core.symbol import Dummy
from sympy.polys.densearith import (dup_mul, dup_mul_ground,
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polytools import named_poly
from sympy.utilities import public
def dup_hermite(n, K):
    """Low-level implementation of Hermite polynomials."""
    if n < 1:
        return [K.one]
    m2, m1 = ([K.one], [K(2), K.zero])
    for i in range(2, n + 1):
        a = dup_lshift(m1, 1, K)
        b = dup_mul_ground(m2, K(i - 1), K)
        m2, m1 = (m1, dup_mul_ground(dup_sub(a, b, K), K(2), K))
    return m1