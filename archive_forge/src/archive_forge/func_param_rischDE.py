import itertools
from functools import reduce
from sympy.core import Dummy, ilcm, Add, Mul, Pow, S
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
from sympy.polys import Poly, lcm, cancel, sqf_list
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.solvers import solve
def param_rischDE(fa, fd, G, DE):
    """
    Solve a Parametric Risch Differential Equation: Dy + f*y == Sum(ci*Gi, (i, 1, m)).

    Explanation
    ===========

    Given a derivation D in k(t), f in k(t), and G
    = [G1, ..., Gm] in k(t)^m, return h = [h1, ..., hr] in k(t)^r and
    a matrix A with m + r columns and entries in Const(k) such that
    Dy + f*y = Sum(ci*Gi, (i, 1, m)) has a solution y
    in k(t) with c1, ..., cm in Const(k) if and only if y = Sum(dj*hj,
    (j, 1, r)) where d1, ..., dr are in Const(k) and (c1, ..., cm,
    d1, ..., dr) is a solution of Ax == 0.

    Elements of k(t) are tuples (a, d) with a and d in k[t].
    """
    m = len(G)
    q, (fa, fd) = weak_normalizer(fa, fd, DE)
    gamma = q
    G = [(q * ga).cancel(gd, include=True) for ga, gd in G]
    a, (ba, bd), G, hn = prde_normal_denom(fa, fd, G, DE)
    gamma *= hn
    A, B, G, hs = prde_special_denom(a, ba, bd, G, DE)
    gamma *= hs
    g = A.gcd(B)
    a, b, g = (A.quo(g), B.quo(g), [gia.cancel(gid * g, include=True) for gia, gid in G])
    q, M = prde_linear_constraints(a, b, g, DE)
    M, _ = constant_system(M, zeros(M.rows, 1, DE.t), DE)
    V = M.nullspace()
    if not V:
        return ([], eye(m, DE.t))
    Mq = Matrix([q])
    r = [(Mq * vj)[0] for vj in V]
    try:
        n = bound_degree(a, b, r, DE, parametric=True)
    except NotImplementedError:
        n = 5
    h, B = param_poly_rischDE(a, b, r, n, DE)
    A = -eye(m, DE.t)
    for vj in V:
        A = A.row_join(vj)
    A = A.row_join(zeros(m, len(h), DE.t))
    A = A.col_join(zeros(B.rows, m, DE.t).row_join(B))
    W = A.nullspace()
    v = len(h)
    shape = (len(W), m + v)
    elements = [wl[:m] + wl[-v:] for wl in W]
    items = [e for row in elements for e in row]
    M = Matrix(*shape, items, DE.t)
    N = M.nullspace()
    C = Matrix([ni[:] for ni in N], DE.t)
    return ([hk.cancel(gamma, include=True) for hk in h], C)