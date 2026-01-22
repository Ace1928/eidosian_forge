from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def gf_Qbasis(Q, p, K):
    """
    Compute a basis of the kernel of ``Q``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_Qmatrix, gf_Qbasis

    >>> gf_Qbasis(gf_Qmatrix([1, 0, 0, 0, 1], 5, ZZ), 5, ZZ)
    [[1, 0, 0, 0], [0, 0, 1, 0]]

    >>> gf_Qbasis(gf_Qmatrix([3, 2, 4], 5, ZZ), 5, ZZ)
    [[1, 0]]

    """
    Q, n = ([list(q) for q in Q], len(Q))
    for k in range(0, n):
        Q[k][k] = (Q[k][k] - K.one) % p
    for k in range(0, n):
        for i in range(k, n):
            if Q[k][i]:
                break
        else:
            continue
        inv = K.invert(Q[k][i], p)
        for j in range(0, n):
            Q[j][i] = Q[j][i] * inv % p
        for j in range(0, n):
            t = Q[j][k]
            Q[j][k] = Q[j][i]
            Q[j][i] = t
        for i in range(0, n):
            if i != k:
                q = Q[k][i]
                for j in range(0, n):
                    Q[j][i] = (Q[j][i] - Q[j][k] * q) % p
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                Q[i][j] = (K.one - Q[i][j]) % p
            else:
                Q[i][j] = -Q[i][j] % p
    basis = []
    for q in Q:
        if any(q):
            basis.append(q)
    return basis