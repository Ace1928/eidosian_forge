from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
def dup_root_upper_bound(f, K):
    """Compute the LMQ upper bound for the positive roots of `f`;
       LMQ (Local Max Quadratic) was developed by Akritas-Strzebonski-Vigklas.

    References
    ==========
    .. [1] Alkiviadis G. Akritas: "Linear and Quadratic Complexity Bounds on the
        Values of the Positive Roots of Polynomials"
        Journal of Universal Computer Science, Vol. 15, No. 3, 523-537, 2009.
    """
    n, P = (len(f), [])
    t = n * [K.one]
    if dup_LC(f, K) < 0:
        f = dup_neg(f, K)
    f = list(reversed(f))
    for i in range(0, n):
        if f[i] >= 0:
            continue
        a, QL = (K.log(-f[i], 2), [])
        for j in range(i + 1, n):
            if f[j] <= 0:
                continue
            q = t[j] + a - K.log(f[j], 2)
            QL.append([q // (j - i), j])
        if not QL:
            continue
        q = min(QL)
        t[q[1]] = t[q[1]] + 1
        P.append(q[0])
    if not P:
        return None
    else:
        return K.get_field()(2) ** (max(P) + 1)