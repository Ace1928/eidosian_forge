from sympy.core.symbol import Dummy
from sympy.polys.densearith import (dup_mul, dup_mul_ground,
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polytools import named_poly
from sympy.utilities import public
def dup_jacobi(n, a, b, K):
    """Low-level implementation of Jacobi polynomials."""
    if n < 1:
        return [K.one]
    m2, m1 = ([K.one], [(a + b) / K(2) + K.one, (a - b) / K(2)])
    for i in range(2, n + 1):
        den = K(i) * (a + b + i) * (a + b + K(2) * i - K(2))
        f0 = (a + b + K(2) * i - K.one) * (a * a - b * b) / (K(2) * den)
        f1 = (a + b + K(2) * i - K.one) * (a + b + K(2) * i - K(2)) * (a + b + K(2) * i) / (K(2) * den)
        f2 = (a + i - K.one) * (b + i - K.one) * (a + b + K(2) * i) / den
        p0 = dup_mul_ground(m1, f0, K)
        p1 = dup_mul_ground(dup_lshift(m1, 1, K), f1, K)
        p2 = dup_mul_ground(m2, f2, K)
        m2, m1 = (m1, dup_sub(dup_add(p0, p1, K), p2, K))
    return m1