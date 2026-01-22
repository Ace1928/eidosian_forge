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
def test_pow1():
    assert refine((-1) ** x, Q.even(x)) == 1
    assert refine((-1) ** x, Q.odd(x)) == -1
    assert refine((-2) ** x, Q.even(x)) == 2 ** x
    assert refine(sqrt(x ** 2)) != Abs(x)
    assert refine(sqrt(x ** 2), Q.complex(x)) != Abs(x)
    assert refine(sqrt(x ** 2), Q.real(x)) == Abs(x)
    assert refine(sqrt(x ** 2), Q.positive(x)) == x
    assert refine((x ** 3) ** Rational(1, 3)) != x
    assert refine((x ** 3) ** Rational(1, 3), Q.real(x)) != x
    assert refine((x ** 3) ** Rational(1, 3), Q.positive(x)) == x
    assert refine(sqrt(1 / x), Q.real(x)) != 1 / sqrt(x)
    assert refine(sqrt(1 / x), Q.positive(x)) == 1 / sqrt(x)
    assert refine((-1) ** (x + y), Q.even(x)) == (-1) ** y
    assert refine((-1) ** (x + y + z), Q.odd(x) & Q.odd(z)) == (-1) ** y
    assert refine((-1) ** (x + y + 1), Q.odd(x)) == (-1) ** y
    assert refine((-1) ** (x + y + 2), Q.odd(x)) == (-1) ** (y + 1)
    assert refine((-1) ** (x + 3)) == (-1) ** (x + 1)
    assert refine((-1) ** ((-1) ** x / 2 - S.Half), Q.integer(x)) == (-1) ** x
    assert refine((-1) ** ((-1) ** x / 2 + S.Half), Q.integer(x)) == (-1) ** (x + 1)
    assert refine((-1) ** ((-1) ** x / 2 + 5 * S.Half), Q.integer(x)) == (-1) ** (x + 1)