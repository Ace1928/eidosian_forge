from operator import mul
from functools import reduce
from sympy.core import oo
from sympy.core.symbol import Dummy
from sympy.polys import Poly, gcd, ZZ, cancel
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
def special_denom(a, ba, bd, ca, cd, DE, case='auto'):
    """
    Special part of the denominator.

    Explanation
    ===========

    case is one of {'exp', 'tan', 'primitive'} for the hyperexponential,
    hypertangent, and primitive cases, respectively.  For the
    hyperexponential (resp. hypertangent) case, given a derivation D on
    k[t] and a in k[t], b, c, in k<t> with Dt/t in k (resp. Dt/(t**2 + 1) in
    k, sqrt(-1) not in k), a != 0, and gcd(a, t) == 1 (resp.
    gcd(a, t**2 + 1) == 1), return the quadruplet (A, B, C, 1/h) such that
    A, B, C, h in k[t] and for any solution q in k<t> of a*Dq + b*q == c,
    r = qh in k[t] satisfies A*Dr + B*r == C.

    For ``case == 'primitive'``, k<t> == k[t], so it returns (a, b, c, 1) in
    this case.

    This constitutes step 2 of the outline given in the rde.py docstring.
    """
    if case == 'auto':
        case = DE.case
    if case == 'exp':
        p = Poly(DE.t, DE.t)
    elif case == 'tan':
        p = Poly(DE.t ** 2 + 1, DE.t)
    elif case in ('primitive', 'base'):
        B = ba.to_field().quo(bd)
        C = ca.to_field().quo(cd)
        return (a, B, C, Poly(1, DE.t))
    else:
        raise ValueError("case must be one of {'exp', 'tan', 'primitive', 'base'}, not %s." % case)
    nb = order_at(ba, p, DE.t) - order_at(bd, p, DE.t)
    nc = order_at(ca, p, DE.t) - order_at(cd, p, DE.t)
    n = min(0, nc - min(0, nb))
    if not nb:
        from .prde import parametric_log_deriv
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
                alphaa, alphad = frac_in(im(-ba.eval(sqrt(-1)) / bd.eval(sqrt(-1)) / a.eval(sqrt(-1))), DE.t)
                betaa, betad = frac_in(re(-ba.eval(sqrt(-1)) / bd.eval(sqrt(-1)) / a.eval(sqrt(-1))), DE.t)
                etaa, etad = frac_in(dcoeff, DE.t)
                if recognize_log_derivative(Poly(2, DE.t) * betaa, betad, DE):
                    A = parametric_log_deriv(alphaa * Poly(sqrt(-1), DE.t) * betad + alphad * betaa, alphad * betad, etaa, etad, DE)
                    if A is not None:
                        Q, m, z = A
                        if Q == 1:
                            n = min(n, m)
    N = max(0, -nb, n - nc)
    pN = p ** N
    pn = p ** (-n)
    A = a * pN
    B = ba * pN.quo(bd) + Poly(n, DE.t) * a * derivation(p, DE).quo(p) * pN
    C = (ca * pN * pn).quo(cd)
    h = pn
    return (A, B, C, h)