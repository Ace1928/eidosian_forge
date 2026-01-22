from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
def dmp_sqf_list(f, u, K, all=False):
    """
    Return square-free decomposition of a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x**5 + 2*x**4*y + x**3*y**2

    >>> R.dmp_sqf_list(f)
    (1, [(x + y, 2), (x, 3)])
    >>> R.dmp_sqf_list(f, all=True)
    (1, [(1, 1), (x + y, 2), (x, 3)])

    """
    if not u:
        return dup_sqf_list(f, K, all=all)
    if K.is_FiniteField:
        return dmp_gf_sqf_list(f, u, K, all=all)
    if K.is_Field:
        coeff = dmp_ground_LC(f, u, K)
        f = dmp_ground_monic(f, u, K)
    else:
        coeff, f = dmp_ground_primitive(f, u, K)
        if K.is_negative(dmp_ground_LC(f, u, K)):
            f = dmp_neg(f, u, K)
            coeff = -coeff
    if dmp_degree(f, u) <= 0:
        return (coeff, [])
    result, i = ([], 1)
    h = dmp_diff(f, 1, u, K)
    g, p, q = dmp_inner_gcd(f, h, u, K)
    while True:
        d = dmp_diff(p, 1, u, K)
        h = dmp_sub(q, d, u, K)
        if dmp_zero_p(h, u):
            result.append((p, i))
            break
        g, p, q = dmp_inner_gcd(p, h, u, K)
        if all or dmp_degree(g, u) > 0:
            result.append((g, i))
        i += 1
    return (coeff, result)