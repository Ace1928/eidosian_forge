from sympy.core import Add, Mul, Symbol, sympify, Dummy, symbols
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.ntheory import nextprime
from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.domains import ZZ
from sympy.polys.factortools import dup_zz_cyclotomic_poly
from sympy.polys.polyclasses import DMP
from sympy.polys.polytools import Poly, PurePoly
from sympy.polys.polyutils import _analyze_gens
from sympy.utilities import subsets, public, filldedent
from sympy.polys.rings import ring
@public
def interpolating_poly(n, x, X='x', Y='y'):
    """Construct Lagrange interpolating polynomial for ``n``
    data points. If a sequence of values are given for ``X`` and ``Y``
    then the first ``n`` values will be used.
    """
    ok = getattr(x, 'free_symbols', None)
    if isinstance(X, str):
        X = symbols('%s:%s' % (X, n))
    elif ok and ok & Tuple(*X).free_symbols:
        ok = False
    if isinstance(Y, str):
        Y = symbols('%s:%s' % (Y, n))
    elif ok and ok & Tuple(*Y).free_symbols:
        ok = False
    if not ok:
        raise ValueError(filldedent('\n            Expecting symbol for x that does not appear in X or Y.\n            Use `interpolate(list(zip(X, Y)), x)` instead.'))
    coeffs = []
    numert = Mul(*[x - X[i] for i in range(n)])
    for i in range(n):
        numer = numert / (x - X[i])
        denom = Mul(*[X[i] - X[j] for j in range(n) if i != j])
        coeffs.append(numer / denom)
    return Add(*[coeff * y for coeff, y in zip(coeffs, Y)])