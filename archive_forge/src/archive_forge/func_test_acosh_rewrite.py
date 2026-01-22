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
def test_acosh_rewrite():
    x = Symbol('x')
    assert acosh(x).rewrite(log) == log(x + sqrt(x - 1) * sqrt(x + 1))
    assert acosh(x).rewrite(asin) == sqrt(x - 1) * (-asin(x) + pi / 2) / sqrt(1 - x)
    assert acosh(x).rewrite(asinh) == sqrt(x - 1) * (-asin(x) + pi / 2) / sqrt(1 - x)
    assert acosh(x).rewrite(atanh) == sqrt(x - 1) * sqrt(x + 1) * atanh(sqrt(x ** 2 - 1) / x) / sqrt(x ** 2 - 1) + pi * sqrt(x - 1) * (-x * sqrt(x ** (-2)) + 1) / (2 * sqrt(1 - x))
    x = Symbol('x', positive=True)
    assert acosh(x).rewrite(atanh) == sqrt(x - 1) * sqrt(x + 1) * atanh(sqrt(x ** 2 - 1) / x) / sqrt(x ** 2 - 1)