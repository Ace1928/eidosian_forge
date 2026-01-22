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
def test_cosh_nonnegative():
    k = symbols('k', real=True)
    n = symbols('n', integer=True)
    assert cosh(k, evaluate=False).is_nonnegative is True
    assert cosh(k + 2 * n * pi * I, evaluate=False).is_nonnegative is True
    assert cosh(I * pi / 4, evaluate=False).is_nonnegative is True
    assert cosh(3 * I * pi / 4, evaluate=False).is_nonnegative is False
    assert cosh(S.Zero, evaluate=False).is_nonnegative is True