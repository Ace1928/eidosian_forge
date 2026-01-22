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
def test_get_subNO():
    p, q, r = symbols('p,q,r')
    assert NO(F(p) * F(q) * F(r)).get_subNO(1) == NO(F(p) * F(r))
    assert NO(F(p) * F(q) * F(r)).get_subNO(0) == NO(F(q) * F(r))
    assert NO(F(p) * F(q) * F(r)).get_subNO(2) == NO(F(p) * F(q))