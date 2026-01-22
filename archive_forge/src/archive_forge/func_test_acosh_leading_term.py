from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.function import (expand_mul, expand_trig)
from sympy.core.numbers import (E, I, Integer, Rational, nan, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, acoth, acsch, asech, asinh, atanh, cosh, coth, csch, sech, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, cos, cot, sec, sin, tan)
from sympy.series.order import O
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.testing.pytest import raises
def test_acosh_leading_term():
    x = Symbol('x')
    assert acosh(x).as_leading_term(x) == I * pi / 2
    assert acosh(x + 1).as_leading_term(x) == sqrt(2) * sqrt(x)
    assert acosh(x - 1).as_leading_term(x) == I * pi
    assert acosh(1 / x).as_leading_term(x, cdir=1) == -log(x) + log(2)
    assert acosh(1 / x).as_leading_term(x, cdir=-1) == -log(x) + log(2) + 2 * I * pi
    assert acosh(I * x - 2).as_leading_term(x, cdir=1) == acosh(-2)
    assert acosh(-I * x - 2).as_leading_term(x, cdir=1) == -2 * I * pi + acosh(-2)
    assert acosh(x ** 2 - I * x + S(1) / 3).as_leading_term(x, cdir=1) == -acosh(S(1) / 3)
    assert acosh(x ** 2 - I * x + S(1) / 3).as_leading_term(x, cdir=-1) == acosh(S(1) / 3)
    assert acosh(1 / (I * x - 3)).as_leading_term(x, cdir=1) == -acosh(-S(1) / 3)
    assert acosh(1 / (I * x - 3)).as_leading_term(x, cdir=-1) == acosh(-S(1) / 3)
    assert acosh(-I * x ** 2 + x - 2).as_leading_term(x, cdir=1) == log(sqrt(3) + 2) - I * pi
    assert acosh(-I * x ** 2 + x - 2).as_leading_term(x, cdir=-1) == log(sqrt(3) + 2) - I * pi