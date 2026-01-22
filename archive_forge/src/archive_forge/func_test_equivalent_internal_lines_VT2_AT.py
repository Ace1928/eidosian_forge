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
def test_equivalent_internal_lines_VT2_AT():
    i, j, k, l = symbols('i j k l', below_fermi=True, cls=Dummy)
    a, b, c, d = symbols('a b c d', above_fermi=True, cls=Dummy)
    exprs = [atv(i, j, a, b) * att(a, b, i, j), atv(j, i, a, b) * att(a, b, i, j), atv(i, j, b, a) * att(a, b, i, j)]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)
    exprs = [atv(i, j, a, b) * att(a, b, i, j), atv(i, j, a, b) * att(b, a, i, j), atv(i, j, a, b) * att(a, b, j, i)]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)
    exprs = [atv(i, j, a, b) * att(a, b, i, j), atv(j, i, a, b) * att(a, b, j, i), atv(i, j, b, a) * att(b, a, i, j), atv(j, i, b, a) * att(b, a, j, i)]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)