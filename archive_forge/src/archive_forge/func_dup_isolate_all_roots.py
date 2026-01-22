from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
def dup_isolate_all_roots(f, K, eps=None, inf=None, sup=None, fast=False):
    """Isolate real and complex roots of a non-square-free polynomial ``f``. """
    if not K.is_ZZ and (not K.is_QQ):
        raise DomainError('isolation of real and complex roots is not supported over %s' % K)
    _, factors = dup_sqf_list(f, K)
    if len(factors) == 1:
        (f, k), = factors
        real_part, complex_part = dup_isolate_all_roots_sqf(f, K, eps=eps, inf=inf, sup=sup, fast=fast)
        real_part = [((a, b), k) for a, b in real_part]
        complex_part = [((a, b), k) for a, b in complex_part]
        return (real_part, complex_part)
    else:
        raise NotImplementedError('only trivial square-free polynomials are supported')