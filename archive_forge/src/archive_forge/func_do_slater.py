from collections import defaultdict
from itertools import product
from functools import reduce
from math import prod
from sympy import SYMPY_DEBUG
from sympy.core import (S, Dummy, symbols, sympify, Tuple, expand, I, pi, Mul,
from sympy.core.mod import Mod
from sympy.core.sorting import default_sort_key
from sympy.functions import (exp, sqrt, root, log, lowergamma, cos,
from sympy.functions.elementary.complexes import polarify, unpolarify
from sympy.functions.special.hyper import (hyper, HyperRep_atanh,
from sympy.matrices import Matrix, eye, zeros
from sympy.polys import apart, poly, Poly
from sympy.series import residue
from sympy.simplify.powsimp import powdenest
from sympy.utilities.iterables import sift
def do_slater(an, bm, ap, bq, z, zfinal):
    func = G_Function(an, bm, ap, bq)
    _, pbm, pap, _ = func.compute_buckets()
    if not can_do(pbm, pap):
        return (S.Zero, False)
    cond = len(an) + len(ap) < len(bm) + len(bq)
    if len(an) + len(ap) == len(bm) + len(bq):
        cond = abs(z) < 1
    if cond is False:
        return (S.Zero, False)
    res = S.Zero
    for m in pbm:
        if len(pbm[m]) == 1:
            bh = pbm[m][0]
            fac = 1
            bo = list(bm)
            bo.remove(bh)
            for bj in bo:
                fac *= gamma(bj - bh)
            for aj in an:
                fac *= gamma(1 + bh - aj)
            for bj in bq:
                fac /= gamma(1 + bh - bj)
            for aj in ap:
                fac /= gamma(aj - bh)
            nap = [1 + bh - a for a in list(an) + list(ap)]
            nbq = [1 + bh - b for b in list(bo) + list(bq)]
            k = polar_lift(S.NegativeOne ** (len(ap) - len(bm)))
            harg = k * zfinal
            premult = (t / k) ** bh
            hyp = _hyperexpand(Hyper_Function(nap, nbq), harg, ops, t, premult, bh, rewrite=None)
            res += fac * hyp
        else:
            b_ = pbm[m][0]
            ki = [bi - b_ for bi in pbm[m][1:]]
            u = len(ki)
            li = [ai - b_ for ai in pap[m][:u + 1]]
            bo = list(bm)
            for b in pbm[m]:
                bo.remove(b)
            ao = list(ap)
            for a in pap[m][:u]:
                ao.remove(a)
            lu = li[-1]
            di = [l - k for l, k in zip(li, ki)]
            s = Dummy('s')
            integrand = z ** s
            for b in bm:
                if not Mod(b, 1) and b.is_Number:
                    b = int(round(b))
                integrand *= gamma(b - s)
            for a in an:
                integrand *= gamma(1 - a + s)
            for b in bq:
                integrand /= gamma(1 - b + s)
            for a in ap:
                integrand /= gamma(a - s)
            integrand = expand_func(integrand)
            for r in range(int(round(lu))):
                resid = residue(integrand, s, b_ + r)
                resid = apply_operators(resid, ops, lambda f: z * f.diff(z))
                res -= resid
            au = b_ + lu
            k = polar_lift(S.NegativeOne ** (len(ao) + len(bo) + 1))
            harg = k * zfinal
            premult = (t / k) ** au
            nap = [1 + au - a for a in list(an) + list(ap)] + [1]
            nbq = [1 + au - b for b in list(bm) + list(bq)]
            hyp = _hyperexpand(Hyper_Function(nap, nbq), harg, ops, t, premult, au, rewrite=None)
            C = S.NegativeOne ** lu / factorial(lu)
            for i in range(u):
                C *= S.NegativeOne ** di[i] / rf(lu - li[i] + 1, di[i])
            for a in an:
                C *= gamma(1 - a + au)
            for b in bo:
                C *= gamma(b - au)
            for a in ao:
                C /= gamma(a - au)
            for b in bq:
                C /= gamma(1 - b + au)
            res += C * hyp
    return (res, cond)