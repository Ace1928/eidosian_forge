from operator import mul
from functools import reduce
from sympy.core import oo
from sympy.core.symbol import Dummy
from sympy.polys import Poly, gcd, ZZ, cancel
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
def weak_normalizer(a, d, DE, z=None):
    """
    Weak normalization.

    Explanation
    ===========

    Given a derivation D on k[t] and f == a/d in k(t), return q in k[t]
    such that f - Dq/q is weakly normalized with respect to t.

    f in k(t) is said to be "weakly normalized" with respect to t if
    residue_p(f) is not a positive integer for any normal irreducible p
    in k[t] such that f is in R_p (Definition 6.1.1).  If f has an
    elementary integral, this is equivalent to no logarithm of
    integral(f) whose argument depends on t has a positive integer
    coefficient, where the arguments of the logarithms not in k(t) are
    in k[t].

    Returns (q, f - Dq/q)
    """
    z = z or Dummy('z')
    dn, ds = splitfactor(d, DE)
    g = gcd(dn, dn.diff(DE.t))
    d_sqf_part = dn.quo(g)
    d1 = d_sqf_part.quo(gcd(d_sqf_part, g))
    a1, b = gcdex_diophantine(d.quo(d1).as_poly(DE.t), d1.as_poly(DE.t), a.as_poly(DE.t))
    r = (a - Poly(z, DE.t) * derivation(d1, DE)).as_poly(DE.t).resultant(d1.as_poly(DE.t))
    r = Poly(r, z)
    if not r.expr.has(z):
        return (Poly(1, DE.t), (a, d))
    N = [i for i in r.real_roots() if i in ZZ and i > 0]
    q = reduce(mul, [gcd(a - Poly(n, DE.t) * derivation(d1, DE), d1) for n in N], Poly(1, DE.t))
    dq = derivation(q, DE)
    sn = q * a - d * dq
    sd = q * d
    sn, sd = sn.cancel(sd, include=True)
    return (q, (sn, sd))