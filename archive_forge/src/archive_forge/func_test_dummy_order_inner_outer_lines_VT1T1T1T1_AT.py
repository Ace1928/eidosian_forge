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
def test_dummy_order_inner_outer_lines_VT1T1T1T1_AT():
    ii, jj = symbols('i j', below_fermi=True)
    aa, bb = symbols('a b', above_fermi=True)
    k, l = symbols('k l', below_fermi=True, cls=Dummy)
    c, d = symbols('c d', above_fermi=True, cls=Dummy)
    exprs = [atv(k, l, c, d) * att(c, ii) * att(d, jj) * att(aa, k) * att(bb, l), atv(k, l, c, d) * att(c, jj) * att(d, ii) * att(aa, k) * att(bb, l), atv(k, l, c, d) * att(c, ii) * att(d, jj) * att(bb, k) * att(aa, l)]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == -substitute_dummies(permut)
    exprs = [atv(k, l, c, d) * att(c, ii) * att(d, jj) * att(aa, k) * att(bb, l), atv(k, l, c, d) * att(c, jj) * att(d, ii) * att(bb, k) * att(aa, l)]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)