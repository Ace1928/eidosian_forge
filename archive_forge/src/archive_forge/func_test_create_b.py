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
def test_create_b():
    i, j, n, m = symbols('i,j,n,m')
    o = Bd(i)
    assert isinstance(o, CreateBoson)
    o = o.subs(i, j)
    assert o.atoms(Symbol) == {j}
    o = Bd(0)
    assert o.apply_operator(BKet([n])) == sqrt(n + 1) * BKet([n + 1])
    o = Bd(n)
    assert o.apply_operator(BKet([n])) == o * BKet([n])