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
def test_basic_apply():
    n = symbols('n')
    e = B(0) * BKet([n])
    assert apply_operators(e) == sqrt(n) * BKet([n - 1])
    e = Bd(0) * BKet([n])
    assert apply_operators(e) == sqrt(n + 1) * BKet([n + 1])