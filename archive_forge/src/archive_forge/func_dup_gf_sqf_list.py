from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
def dup_gf_sqf_list(f, K, all=False):
    """Compute square-free decomposition of ``f`` in ``GF(p)[x]``. """
    f = dup_convert(f, K, K.dom)
    coeff, factors = gf_sqf_list(f, K.mod, K.dom, all=all)
    for i, (f, k) in enumerate(factors):
        factors[i] = (dup_convert(f, K.dom, K), k)
    return (K.convert(coeff, K.dom), factors)