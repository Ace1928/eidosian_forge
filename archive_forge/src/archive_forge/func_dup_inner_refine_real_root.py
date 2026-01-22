from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
def dup_inner_refine_real_root(f, M, K, eps=None, steps=None, disjoint=None, fast=False, mobius=False):
    """Refine a positive root of `f` given a Mobius transform or an interval. """
    F = K.get_field()
    if len(M) == 2:
        a, b, c, d = _mobius_from_interval(M, F)
    else:
        a, b, c, d = M
    while not c:
        f, (a, b, c, d) = dup_step_refine_real_root(f, (a, b, c, d), K, fast=fast)
    if eps is not None and steps is not None:
        for i in range(0, steps):
            if abs(F(a, c) - F(b, d)) >= eps:
                f, (a, b, c, d) = dup_step_refine_real_root(f, (a, b, c, d), K, fast=fast)
            else:
                break
    else:
        if eps is not None:
            while abs(F(a, c) - F(b, d)) >= eps:
                f, (a, b, c, d) = dup_step_refine_real_root(f, (a, b, c, d), K, fast=fast)
        if steps is not None:
            for i in range(0, steps):
                f, (a, b, c, d) = dup_step_refine_real_root(f, (a, b, c, d), K, fast=fast)
    if disjoint is not None:
        while True:
            u, v = _mobius_to_interval((a, b, c, d), F)
            if v <= disjoint or disjoint <= u:
                break
            else:
                f, (a, b, c, d) = dup_step_refine_real_root(f, (a, b, c, d), K, fast=fast)
    if not mobius:
        return _mobius_to_interval((a, b, c, d), F)
    else:
        return (f, (a, b, c, d))