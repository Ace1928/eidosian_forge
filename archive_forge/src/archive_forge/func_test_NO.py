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
def test_NO():
    i, j, k, l = symbols('i j k l', below_fermi=True)
    a, b, c, d = symbols('a b c d', above_fermi=True)
    p, q, r, s = symbols('p q r s', cls=Dummy)
    assert NO(Fd(p) * F(q) + Fd(a) * F(b)) == NO(Fd(p) * F(q)) + NO(Fd(a) * F(b))
    assert NO(Fd(i) * NO(F(j) * Fd(a))) == NO(Fd(i) * F(j) * Fd(a))
    assert NO(1) == 1
    assert NO(i) == i
    assert NO(Fd(a) * Fd(b) * (F(c) + F(d))) == NO(Fd(a) * Fd(b) * F(c)) + NO(Fd(a) * Fd(b) * F(d))
    assert NO(Fd(a) * F(b))._remove_brackets() == Fd(a) * F(b)
    assert NO(F(j) * Fd(i))._remove_brackets() == F(j) * Fd(i)
    assert NO(Fd(p) * F(q)).subs(Fd(p), Fd(a) + Fd(i)) == NO(Fd(a) * F(q)) + NO(Fd(i) * F(q))
    assert NO(Fd(p) * F(q)).subs(F(q), F(a) + F(i)) == NO(Fd(p) * F(a)) + NO(Fd(p) * F(i))
    expr = NO(Fd(p) * F(q))._remove_brackets()
    assert wicks(expr) == NO(expr)
    assert NO(Fd(a) * F(b)) == -NO(F(b) * Fd(a))
    no = NO(Fd(a) * F(i) * F(b) * Fd(j))
    l1 = list(no.iter_q_creators())
    assert l1 == [0, 1]
    l2 = list(no.iter_q_annihilators())
    assert l2 == [3, 2]
    no = NO(Fd(a) * Fd(i))
    assert no.has_q_creators == 1
    assert no.has_q_annihilators == -1
    assert str(no) == ':CreateFermion(a)*CreateFermion(i):'
    assert repr(no) == 'NO(CreateFermion(a)*CreateFermion(i))'
    assert latex(no) == '\\left\\{{a^\\dagger_{a}} {a^\\dagger_{i}}\\right\\}'
    raises(NotImplementedError, lambda: NO(Bd(p) * F(q)))