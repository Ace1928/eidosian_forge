from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
def dup_step_refine_real_root(f, M, K, fast=False):
    """One step of positive real root refinement algorithm. """
    a, b, c, d = M
    if a == b and c == d:
        return (f, (a, b, c, d))
    A = dup_root_lower_bound(f, K)
    if A is not None:
        A = K(int(A))
    else:
        A = K.zero
    if fast and A > 16:
        f = dup_scale(f, A, K)
        a, c, A = (A * a, A * c, K.one)
    if A >= K.one:
        f = dup_shift(f, A, K)
        b, d = (A * a + b, A * c + d)
        if not dup_eval(f, K.zero, K):
            return (f, (b, b, d, d))
    f, g = (dup_shift(f, K.one, K), f)
    a1, b1, c1, d1 = (a, a + b, c, c + d)
    if not dup_eval(f, K.zero, K):
        return (f, (b1, b1, d1, d1))
    k = dup_sign_variations(f, K)
    if k == 1:
        a, b, c, d = (a1, b1, c1, d1)
    else:
        f = dup_shift(dup_reverse(g), K.one, K)
        if not dup_eval(f, K.zero, K):
            f = dup_rshift(f, 1, K)
        a, b, c, d = (b, a + b, d, c + d)
    return (f, (a, b, c, d))