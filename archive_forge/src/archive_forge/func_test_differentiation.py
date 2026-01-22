from sympy.core import symbols, Symbol, Tuple, oo, Dummy
from sympy.tensor.indexed import IndexException
from sympy.testing.pytest import raises
from sympy.utilities.iterables import iterable
from sympy.concrete.summations import Sum
from sympy.core.function import Function, Subs, Derivative
from sympy.core.relational import (StrictLessThan, GreaterThan,
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.series.order import Order
from sympy.sets.fancysets import Range
from sympy.tensor.indexed import IndexedBase, Idx, Indexed
def test_differentiation():
    from sympy.functions.special.tensor_functions import KroneckerDelta
    i, j, k, l = symbols('i j k l', cls=Idx)
    a = symbols('a')
    m, n = symbols('m, n', integer=True, finite=True)
    assert m.is_real
    h, L = symbols('h L', cls=IndexedBase)
    hi, hj = (h[i], h[j])
    expr = hi
    assert expr.diff(hj) == KroneckerDelta(i, j)
    assert expr.diff(hi) == KroneckerDelta(i, i)
    expr = S(2) * hi
    assert expr.diff(hj) == S(2) * KroneckerDelta(i, j)
    assert expr.diff(hi) == S(2) * KroneckerDelta(i, i)
    assert expr.diff(a) is S.Zero
    assert Sum(expr, (i, -oo, oo)).diff(hj) == Sum(2 * KroneckerDelta(i, j), (i, -oo, oo))
    assert Sum(expr.diff(hj), (i, -oo, oo)) == Sum(2 * KroneckerDelta(i, j), (i, -oo, oo))
    assert Sum(expr, (i, -oo, oo)).diff(hj).doit() == 2
    assert Sum(expr.diff(hi), (i, -oo, oo)).doit() == Sum(2, (i, -oo, oo)).doit()
    assert Sum(expr, (i, -oo, oo)).diff(hi).doit() is oo
    expr = a * hj * hj / S(2)
    assert expr.diff(hi) == a * h[j] * KroneckerDelta(i, j)
    assert expr.diff(a) == hj * hj / S(2)
    assert expr.diff(a, 2) is S.Zero
    assert Sum(expr, (i, -oo, oo)).diff(hi) == Sum(a * KroneckerDelta(i, j) * h[j], (i, -oo, oo))
    assert Sum(expr.diff(hi), (i, -oo, oo)) == Sum(a * KroneckerDelta(i, j) * h[j], (i, -oo, oo))
    assert Sum(expr, (i, -oo, oo)).diff(hi).doit() == a * h[j]
    assert Sum(expr, (j, -oo, oo)).diff(hi) == Sum(a * KroneckerDelta(i, j) * h[j], (j, -oo, oo))
    assert Sum(expr.diff(hi), (j, -oo, oo)) == Sum(a * KroneckerDelta(i, j) * h[j], (j, -oo, oo))
    assert Sum(expr, (j, -oo, oo)).diff(hi).doit() == a * h[i]
    expr = a * sin(hj * hj)
    assert expr.diff(hi) == 2 * a * cos(hj * hj) * hj * KroneckerDelta(i, j)
    assert expr.diff(hj) == 2 * a * cos(hj * hj) * hj
    expr = a * L[i, j] * h[j]
    assert expr.diff(hi) == a * L[i, j] * KroneckerDelta(i, j)
    assert expr.diff(hj) == a * L[i, j]
    assert expr.diff(L[i, j]) == a * h[j]
    assert expr.diff(L[k, l]) == a * KroneckerDelta(i, k) * KroneckerDelta(j, l) * h[j]
    assert expr.diff(L[i, l]) == a * KroneckerDelta(j, l) * h[j]
    assert Sum(expr, (j, -oo, oo)).diff(L[k, l]) == Sum(a * KroneckerDelta(i, k) * KroneckerDelta(j, l) * h[j], (j, -oo, oo))
    assert Sum(expr, (j, -oo, oo)).diff(L[k, l]).doit() == a * KroneckerDelta(i, k) * h[l]
    assert h[m].diff(h[m]) == 1
    assert h[m].diff(h[n]) == KroneckerDelta(m, n)
    assert Sum(a * h[m], (m, -oo, oo)).diff(h[n]) == Sum(a * KroneckerDelta(m, n), (m, -oo, oo))
    assert Sum(a * h[m], (m, -oo, oo)).diff(h[n]).doit() == a
    assert Sum(a * h[m], (n, -oo, oo)).diff(h[n]) == Sum(a * KroneckerDelta(m, n), (n, -oo, oo))
    assert Sum(a * h[m], (m, -oo, oo)).diff(h[m]).doit() == oo * a