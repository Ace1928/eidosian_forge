from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
def dup_sqf_list(f, K, all=False):
    """
    Return square-free decomposition of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> f = 2*x**5 + 16*x**4 + 50*x**3 + 76*x**2 + 56*x + 16

    >>> R.dup_sqf_list(f)
    (2, [(x + 1, 2), (x + 2, 3)])
    >>> R.dup_sqf_list(f, all=True)
    (2, [(1, 1), (x + 1, 2), (x + 2, 3)])

    """
    if K.is_FiniteField:
        return dup_gf_sqf_list(f, K, all=all)
    if K.is_Field:
        coeff = dup_LC(f, K)
        f = dup_monic(f, K)
    else:
        coeff, f = dup_primitive(f, K)
        if K.is_negative(dup_LC(f, K)):
            f = dup_neg(f, K)
            coeff = -coeff
    if dup_degree(f) <= 0:
        return (coeff, [])
    result, i = ([], 1)
    h = dup_diff(f, 1, K)
    g, p, q = dup_inner_gcd(f, h, K)
    while True:
        d = dup_diff(p, 1, K)
        h = dup_sub(q, d, K)
        if not h:
            result.append((p, i))
            break
        g, p, q = dup_inner_gcd(p, h, K)
        if all or dup_degree(g) > 0:
            result.append((g, i))
        i += 1
    return (coeff, result)