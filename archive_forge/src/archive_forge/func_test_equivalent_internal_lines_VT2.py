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
def test_equivalent_internal_lines_VT2():
    i, j, k, l = symbols('i j k l', below_fermi=True, cls=Dummy)
    a, b, c, d = symbols('a b c d', above_fermi=True, cls=Dummy)
    v = Function('v')
    t = Function('t')
    dums = _get_ordered_dummies
    exprs = [v(i, j, a, b) * t(a, b, i, j), v(j, i, a, b) * t(a, b, i, j), v(i, j, b, a) * t(a, b, i, j), v(j, i, b, a) * t(a, b, i, j)]
    for permut in exprs[1:]:
        assert dums(exprs[0]) == dums(permut)
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)
    exprs = [v(i, j, a, b) * t(a, b, i, j), v(i, j, a, b) * t(b, a, i, j), v(i, j, a, b) * t(a, b, j, i), v(i, j, a, b) * t(b, a, j, i)]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) != substitute_dummies(permut)
    exprs = [v(i, j, a, b) * t(a, b, i, j), v(j, i, a, b) * t(a, b, j, i), v(i, j, b, a) * t(b, a, i, j), v(j, i, b, a) * t(b, a, j, i)]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)