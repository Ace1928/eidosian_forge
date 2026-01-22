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
def test_simplifications():
    x = Symbol('x')
    assert sinh(asinh(x)) == x
    assert sinh(acosh(x)) == sqrt(x - 1) * sqrt(x + 1)
    assert sinh(atanh(x)) == x / sqrt(1 - x ** 2)
    assert sinh(acoth(x)) == 1 / (sqrt(x - 1) * sqrt(x + 1))
    assert cosh(asinh(x)) == sqrt(1 + x ** 2)
    assert cosh(acosh(x)) == x
    assert cosh(atanh(x)) == 1 / sqrt(1 - x ** 2)
    assert cosh(acoth(x)) == x / (sqrt(x - 1) * sqrt(x + 1))
    assert tanh(asinh(x)) == x / sqrt(1 + x ** 2)
    assert tanh(acosh(x)) == sqrt(x - 1) * sqrt(x + 1) / x
    assert tanh(atanh(x)) == x
    assert tanh(acoth(x)) == 1 / x
    assert coth(asinh(x)) == sqrt(1 + x ** 2) / x
    assert coth(acosh(x)) == x / (sqrt(x - 1) * sqrt(x + 1))
    assert coth(atanh(x)) == 1 / x
    assert coth(acoth(x)) == x
    assert csch(asinh(x)) == 1 / x
    assert csch(acosh(x)) == 1 / (sqrt(x - 1) * sqrt(x + 1))
    assert csch(atanh(x)) == sqrt(1 - x ** 2) / x
    assert csch(acoth(x)) == sqrt(x - 1) * sqrt(x + 1)
    assert sech(asinh(x)) == 1 / sqrt(1 + x ** 2)
    assert sech(acosh(x)) == 1 / x
    assert sech(atanh(x)) == sqrt(1 - x ** 2)
    assert sech(acoth(x)) == sqrt(x - 1) * sqrt(x + 1) / x