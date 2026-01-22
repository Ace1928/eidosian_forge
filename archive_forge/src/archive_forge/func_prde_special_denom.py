import itertools
from functools import reduce
from sympy.core import Dummy, ilcm, Add, Mul, Pow, S
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
from sympy.polys import Poly, lcm, cancel, sqf_list
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.solvers import solve
def prde_special_denom(a, ba, bd, G, DE, case='auto'):
    """
    Parametric Risch Differential Equation - Special part of the denominator.

    Explanation
    ===========

    Case is one of {'exp', 'tan', 'primitive'} for the hyperexponential,
    hypertangent, and primitive cases, respectively.  For the hyperexponential
    (resp. hypertangent) case, given a derivation D on k[t] and a in k[t],
    b in k<t>, and g1, ..., gm in k(t) with Dt/t in k (resp. Dt/(t**2 + 1) in
    k, sqrt(-1) not in k), a != 0, and gcd(a, t) == 1 (resp.
    gcd(a, t**2 + 1) == 1), return the tuple (A, B, GG, h) such that A, B, h in
    k[t], GG = [gg1, ..., ggm] in k(t)^m, and for any solution c1, ..., cm in
    Const(k) and q in k<t> of a*Dq + b*q == Sum(ci*gi, (i, 1, m)), r == q*h in
    k[t] satisfies A*Dr + B*r == Sum(ci*ggi, (i, 1, m)).

    For case == 'primitive', k<t> == k[t], so it returns (a, b, G, 1) in this
    case.
    """
    if case == 'auto':
        case = DE.case
    if case == 'exp':
        p = Poly(DE.t, DE.t)
    elif case == 'tan':
        p = Poly(DE.t ** 2 + 1, DE.t)
    elif case in ('primitive', 'base'):
        B = ba.quo(bd)
        return (a, B, G, Poly(1, DE.t))
    else:
        raise ValueError("case must be one of {'exp', 'tan', 'primitive', 'base'}, not %s." % case)
    nb = order_at(ba, p, DE.t) - order_at(bd, p, DE.t)
    nc = min([order_at(Ga, p, DE.t) - order_at(Gd, p, DE.t) for Ga, Gd in G])
    n = min(0, nc - min(0, nb))
    if not nb:
        if case == 'exp':
            dcoeff = DE.d.quo(Poly(DE.t, DE.t))
            with DecrementLevel(DE):
                alphaa, alphad = frac_in(-ba.eval(0) / bd.eval(0) / a.eval(0), DE.t)
                etaa, etad = frac_in(dcoeff, DE.t)
                A = parametric_log_deriv(alphaa, alphad, etaa, etad, DE)
                if A is not None:
                    Q, m, z = A
                    if Q == 1:
                        n = min(n, m)
        elif case == 'tan':
            dcoeff = DE.d.quo(Poly(DE.t ** 2 + 1, DE.t))
            with DecrementLevel(DE):
                betaa, alphaa, alphad = real_imag(ba, bd * a, DE.t)
                betad = alphad
                etaa, etad = frac_in(dcoeff, DE.t)
                if recognize_log_derivative(Poly(2, DE.t) * betaa, betad, DE):
                    A = parametric_log_deriv(alphaa, alphad, etaa, etad, DE)
                    B = parametric_log_deriv(betaa, betad, etaa, etad, DE)
                    if A is not None and B is not None:
                        Q, s, z = A
                        if Q == 1:
                            n = min(n, s / 2)
    N = max(0, -nb)
    pN = p ** N
    pn = p ** (-n)
    A = a * pN
    B = ba * pN.quo(bd) + Poly(n, DE.t) * a * derivation(p, DE).quo(p) * pN
    G = [(Ga * pN * pn).cancel(Gd, include=True) for Ga, Gd in G]
    h = pn
    return (A, B, G, h)