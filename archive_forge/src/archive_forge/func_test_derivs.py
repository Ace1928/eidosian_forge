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
def test_derivs():
    x = Symbol('x')
    assert coth(x).diff(x) == -sinh(x) ** (-2)
    assert sinh(x).diff(x) == cosh(x)
    assert cosh(x).diff(x) == sinh(x)
    assert tanh(x).diff(x) == -tanh(x) ** 2 + 1
    assert csch(x).diff(x) == -coth(x) * csch(x)
    assert sech(x).diff(x) == -tanh(x) * sech(x)
    assert acoth(x).diff(x) == 1 / (-x ** 2 + 1)
    assert asinh(x).diff(x) == 1 / sqrt(x ** 2 + 1)
    assert acosh(x).diff(x) == 1 / (sqrt(x - 1) * sqrt(x + 1))
    assert acosh(x).diff(x) == acosh(x).rewrite(log).diff(x).together()
    assert atanh(x).diff(x) == 1 / (-x ** 2 + 1)
    assert asech(x).diff(x) == -1 / (x * sqrt(1 - x ** 2))
    assert acsch(x).diff(x) == -1 / (x ** 2 * sqrt(1 + x ** (-2)))