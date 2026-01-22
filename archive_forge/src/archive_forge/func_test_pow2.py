from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine
from sympy.core.expr import Expr
from sympy.core.numbers import (I, Rational, nan, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import (Abs, arg, im, re, sign)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan, atan2)
from sympy.abc import w, x, y, z
from sympy.core.relational import Eq, Ne
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices.expressions.matexpr import MatrixSymbol
def test_pow2():
    assert refine((-1) ** ((-1) ** x / 2 - 7 * S.Half), Q.integer(x)) == (-1) ** (x + 1)
    assert refine((-1) ** ((-1) ** x / 2 - 9 * S.Half), Q.integer(x)) == (-1) ** x
    assert refine(Abs(x) ** 2, Q.real(x)) == x ** 2
    assert refine(Abs(x) ** 3, Q.real(x)) == Abs(x) ** 3
    assert refine(Abs(x) ** 2) == Abs(x) ** 2