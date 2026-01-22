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
def test_equivalent_internal_lines_VT1T1_AT():
    i, j, k, l = symbols('i j k l', below_fermi=True, cls=Dummy)
    a, b, c, d = symbols('a b c d', above_fermi=True, cls=Dummy)
    exprs = [atv(i, j, a, b) * att(a, i) * att(b, j), atv(j, i, a, b) * att(a, i) * att(b, j), atv(i, j, b, a) * att(a, i) * att(b, j)]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)
    exprs = [atv(i, j, a, b) * att(a, i) * att(b, j), atv(j, i, b, a) * att(a, i) * att(b, j)]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)
    exprs = [atv(i, j, a, b) * att(a, i) * att(b, j), atv(i, j, a, b) * att(b, i) * att(a, j)]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)
    exprs = [atv(i, j, a, b) * att(a, i) * att(b, j), atv(j, i, a, b) * att(a, j) * att(b, i), atv(i, j, b, a) * att(b, i) * att(a, j), atv(j, i, b, a) * att(b, j) * att(a, i)]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)