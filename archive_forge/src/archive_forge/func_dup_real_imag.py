from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (
from sympy.utilities import variations
from math import ceil as _ceil, log as _log
def dup_real_imag(f, K):
    """
    Return bivariate polynomials ``f1`` and ``f2``, such that ``f = f1 + f2*I``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dup_real_imag(x**3 + x**2 + x + 1)
    (x**3 + x**2 - 3*x*y**2 + x - y**2 + 1, 3*x**2*y + 2*x*y - y**3 + y)

    """
    if not K.is_ZZ and (not K.is_QQ):
        raise DomainError('computing real and imaginary parts is not supported over %s' % K)
    f1 = dmp_zero(1)
    f2 = dmp_zero(1)
    if not f:
        return (f1, f2)
    g = [[[K.one, K.zero]], [[K.one], []]]
    h = dmp_ground(f[0], 2)
    for c in f[1:]:
        h = dmp_mul(h, g, 2, K)
        h = dmp_add_term(h, dmp_ground(c, 1), 0, 2, K)
    H = dup_to_raw_dict(h)
    for k, h in H.items():
        m = k % 4
        if not m:
            f1 = dmp_add(f1, h, 1, K)
        elif m == 1:
            f2 = dmp_add(f2, h, 1, K)
        elif m == 2:
            f1 = dmp_sub(f1, h, 1, K)
        else:
            f2 = dmp_sub(f2, h, 1, K)
    return (f1, f2)