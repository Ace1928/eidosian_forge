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
def test_dummy_order_well_defined():
    aa, bb = symbols('a b', above_fermi=True)
    k, l, m = symbols('k l m', below_fermi=True, cls=Dummy)
    c, d = symbols('c d', above_fermi=True, cls=Dummy)
    p, q = symbols('p q', cls=Dummy)
    A = Function('A')
    B = Function('B')
    C = Function('C')
    dums = _get_ordered_dummies
    assert dums(A(k, l) * B(l, k)) == [k, l]
    assert dums(A(l, k) * B(l, k)) == [l, k]
    assert dums(A(k, l) * B(k, l)) == [k, l]
    assert dums(A(l, k) * B(k, l)) == [l, k]
    assert dums(A(k, l) * B(l, m) * C(k, m)) == [l, k, m]
    assert dums(A(k, l) * B(l, m) * C(m, k)) == [l, k, m]
    assert dums(A(l, k) * B(l, m) * C(k, m)) == [l, k, m]
    assert dums(A(l, k) * B(l, m) * C(m, k)) == [l, k, m]
    assert dums(A(k, l) * B(m, l) * C(k, m)) == [l, k, m]
    assert dums(A(k, l) * B(m, l) * C(m, k)) == [l, k, m]
    assert dums(A(l, k) * B(m, l) * C(k, m)) == [l, k, m]
    assert dums(A(l, k) * B(m, l) * C(m, k)) == [l, k, m]
    assert dums(A(k, aa, l) * A(l, bb, m) * A(bb, k, m)) == [l, k, m]
    assert dums(A(k, aa, l) * A(l, bb, m) * A(bb, m, k)) == [l, k, m]
    assert dums(A(k, aa, l) * A(m, bb, l) * A(bb, k, m)) == [l, k, m]
    assert dums(A(k, aa, l) * A(m, bb, l) * A(bb, m, k)) == [l, k, m]
    assert dums(A(l, aa, k) * A(l, bb, m) * A(bb, k, m)) == [l, k, m]
    assert dums(A(l, aa, k) * A(l, bb, m) * A(bb, m, k)) == [l, k, m]
    assert dums(A(l, aa, k) * A(m, bb, l) * A(bb, k, m)) == [l, k, m]
    assert dums(A(l, aa, k) * A(m, bb, l) * A(bb, m, k)) == [l, k, m]
    assert dums(A(p, c, k) * B(p, c, k)) == [k, c, p]
    assert dums(A(p, k, c) * B(p, c, k)) == [k, c, p]
    assert dums(A(c, k, p) * B(p, c, k)) == [k, c, p]
    assert dums(A(c, p, k) * B(p, c, k)) == [k, c, p]
    assert dums(A(k, c, p) * B(p, c, k)) == [k, c, p]
    assert dums(A(k, p, c) * B(p, c, k)) == [k, c, p]
    assert dums(B(p, c, k) * A(p, c, k)) == [k, c, p]
    assert dums(B(p, k, c) * A(p, c, k)) == [k, c, p]
    assert dums(B(c, k, p) * A(p, c, k)) == [k, c, p]
    assert dums(B(c, p, k) * A(p, c, k)) == [k, c, p]
    assert dums(B(k, c, p) * A(p, c, k)) == [k, c, p]
    assert dums(B(k, p, c) * A(p, c, k)) == [k, c, p]