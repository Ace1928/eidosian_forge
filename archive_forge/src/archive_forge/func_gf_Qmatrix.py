from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def gf_Qmatrix(f, p, K):
    """
    Calculate Berlekamp's ``Q`` matrix.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_Qmatrix

    >>> gf_Qmatrix([3, 2, 4], 5, ZZ)
    [[1, 0],
     [3, 4]]

    >>> gf_Qmatrix([1, 0, 0, 0, 1], 5, ZZ)
    [[1, 0, 0, 0],
     [0, 4, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 4]]

    """
    n, r = (gf_degree(f), int(p))
    q = [K.one] + [K.zero] * (n - 1)
    Q = [list(q)] + [[]] * (n - 1)
    for i in range(1, (n - 1) * r + 1):
        qq, c = ([-q[-1] * f[-1] % p], q[-1])
        for j in range(1, n):
            qq.append((q[j - 1] - c * f[-j - 1]) % p)
        if not i % r:
            Q[i // r] = list(qq)
        q = qq
    return Q