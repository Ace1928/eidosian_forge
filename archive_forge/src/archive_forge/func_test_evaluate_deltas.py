from sympy.physics.secondquant import (
from sympy.concrete.summations import Sum
from sympy.core.function import (Function, expand)
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.printing.repr import srepr
from sympy.simplify.simplify import simplify
from sympy.testing.pytest import slow, raises
from sympy.printing.latex import latex
def test_evaluate_deltas():
    i, j, k = symbols('i,j,k')
    r = KroneckerDelta(i, j) * KroneckerDelta(j, k)
    assert evaluate_deltas(r) == KroneckerDelta(i, k)
    r = KroneckerDelta(i, 0) * KroneckerDelta(j, k)
    assert evaluate_deltas(r) == KroneckerDelta(i, 0) * KroneckerDelta(j, k)
    r = KroneckerDelta(1, j) * KroneckerDelta(j, k)
    assert evaluate_deltas(r) == KroneckerDelta(1, k)
    r = KroneckerDelta(j, 2) * KroneckerDelta(k, j)
    assert evaluate_deltas(r) == KroneckerDelta(2, k)
    r = KroneckerDelta(i, 0) * KroneckerDelta(i, j) * KroneckerDelta(j, 1)
    assert evaluate_deltas(r) == 0
    r = KroneckerDelta(0, i) * KroneckerDelta(0, j) * KroneckerDelta(1, j) * KroneckerDelta(1, j)
    assert evaluate_deltas(r) == 0