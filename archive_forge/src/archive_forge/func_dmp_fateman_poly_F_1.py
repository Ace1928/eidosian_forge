from sympy.core import Add, Mul, Symbol, sympify, Dummy, symbols
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.ntheory import nextprime
from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.domains import ZZ
from sympy.polys.factortools import dup_zz_cyclotomic_poly
from sympy.polys.polyclasses import DMP
from sympy.polys.polytools import Poly, PurePoly
from sympy.polys.polyutils import _analyze_gens
from sympy.utilities import subsets, public, filldedent
from sympy.polys.rings import ring
def dmp_fateman_poly_F_1(n, K):
    """Fateman's GCD benchmark: trivial GCD """
    u = [K(1), K(0)]
    for i in range(n):
        u = [dmp_one(i, K), u]
    v = [K(1), K(0), K(0)]
    for i in range(0, n):
        v = [dmp_one(i, K), dmp_zero(i), v]
    m = n - 1
    U = dmp_add_term(u, dmp_ground(K(1), m), 0, n, K)
    V = dmp_add_term(u, dmp_ground(K(2), m), 0, n, K)
    f = [[-K(3), K(0)], [], [K(1), K(0), -K(1)]]
    W = dmp_add_term(v, dmp_ground(K(1), m), 0, n, K)
    Y = dmp_raise(f, m, 1, K)
    F = dmp_mul(U, V, n, K)
    G = dmp_mul(W, Y, n, K)
    H = dmp_one(n, K)
    return (F, G, H)