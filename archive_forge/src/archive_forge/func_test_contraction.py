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
def test_contraction():
    i, j, k, l = symbols('i,j,k,l', below_fermi=True)
    a, b, c, d = symbols('a,b,c,d', above_fermi=True)
    p, q, r, s = symbols('p,q,r,s')
    assert contraction(Fd(i), F(j)) == KroneckerDelta(i, j)
    assert contraction(F(a), Fd(b)) == KroneckerDelta(a, b)
    assert contraction(F(a), Fd(i)) == 0
    assert contraction(Fd(a), F(i)) == 0
    assert contraction(F(i), Fd(a)) == 0
    assert contraction(Fd(i), F(a)) == 0
    assert contraction(Fd(i), F(p)) == KroneckerDelta(i, p)
    restr = evaluate_deltas(contraction(Fd(p), F(q)))
    assert restr.is_only_below_fermi
    restr = evaluate_deltas(contraction(F(p), Fd(q)))
    assert restr.is_only_above_fermi
    raises(ContractionAppliesOnlyToFermions, lambda: contraction(B(a), Fd(b)))