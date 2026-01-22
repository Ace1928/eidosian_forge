import itertools
from functools import reduce
from sympy.core import Dummy, ilcm, Add, Mul, Pow, S
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
from sympy.polys import Poly, lcm, cancel, sqf_list
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.solvers import solve
def limited_integrate(fa, fd, G, DE):
    """
    Solves the limited integration problem:  f = Dv + Sum(ci*wi, (i, 1, n))
    """
    fa, fd = (fa * Poly(1 / fd.LC(), DE.t), fd.monic())
    Fa = Poly(0, DE.t)
    Fd = Poly(1, DE.t)
    G = [(fa, fd)] + G
    h, A = param_rischDE(Fa, Fd, G, DE)
    V = A.nullspace()
    V = [v for v in V if v[0] != 0]
    if not V:
        return None
    else:
        c0 = V[0][0]
        v = V[0] / -c0
        r = len(h)
        m = len(v) - r - 1
        C = list(v[1:m + 1])
        y = -sum([v[m + 1 + i] * h[i][0].as_expr() / h[i][1].as_expr() for i in range(r)])
        y_num, y_den = y.as_numer_denom()
        Ya, Yd = (Poly(y_num, DE.t), Poly(y_den, DE.t))
        Y = (Ya * Poly(1 / Yd.LC(), DE.t), Yd.monic())
        return (Y, C)