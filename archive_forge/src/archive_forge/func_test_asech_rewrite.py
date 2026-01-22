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
def test_asech_rewrite():
    x = Symbol('x')
    assert asech(x).rewrite(log) == log(1 / x + sqrt(1 / x - 1) * sqrt(1 / x + 1))
    assert asech(x).rewrite(acosh) == acosh(1 / x)
    assert asech(x).rewrite(asinh) == sqrt(-1 + 1 / x) * (-asin(1 / x) + pi / 2) / sqrt(1 - 1 / x)
    assert asech(x).rewrite(atanh) == sqrt(x + 1) * sqrt(1 / (x + 1)) * atanh(sqrt(1 - x ** 2)) + I * pi * (-sqrt(x) * sqrt(1 / x) + 1 - I * sqrt(x ** 2) / (2 * sqrt(-x ** 2)) - I * sqrt(-x) / (2 * sqrt(x)))