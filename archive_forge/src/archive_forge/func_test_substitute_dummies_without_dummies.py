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
def test_substitute_dummies_without_dummies():
    i, j = symbols('i,j')
    assert substitute_dummies(att(i, j) + 2) == att(i, j) + 2
    assert substitute_dummies(att(i, j) + 1) == att(i, j) + 1