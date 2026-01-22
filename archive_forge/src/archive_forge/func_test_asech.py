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
def test_asech():
    x = Symbol('x')
    assert unchanged(asech, -x)
    assert asech(1) == 0
    assert asech(-1) == pi * I
    assert asech(0) is oo
    assert asech(2) == I * pi / 3
    assert asech(-2) == 2 * I * pi / 3
    assert asech(nan) is nan
    assert asech(oo) == I * pi / 2
    assert asech(-oo) == I * pi / 2
    assert asech(zoo) == I * AccumBounds(-pi / 2, pi / 2)
    assert asech(I) == log(1 + sqrt(2)) - I * pi / 2
    assert asech(-I) == log(1 + sqrt(2)) + I * pi / 2
    assert asech(sqrt(2) - sqrt(6)) == 11 * I * pi / 12
    assert asech(sqrt(2 - 2 / sqrt(5))) == I * pi / 10
    assert asech(-sqrt(2 - 2 / sqrt(5))) == 9 * I * pi / 10
    assert asech(2 / sqrt(2 + sqrt(2))) == I * pi / 8
    assert asech(-2 / sqrt(2 + sqrt(2))) == 7 * I * pi / 8
    assert asech(sqrt(5) - 1) == I * pi / 5
    assert asech(1 - sqrt(5)) == 4 * I * pi / 5
    assert asech(-sqrt(2 * (2 + sqrt(2)))) == 5 * I * pi / 8
    assert asech(sqrt(2)) == acosh(1 / sqrt(2))
    assert asech(2 / sqrt(3)) == acosh(sqrt(3) / 2)
    assert asech(2 / sqrt(2 + sqrt(2))) == acosh(sqrt(2 + sqrt(2)) / 2)
    assert asech(2) == acosh(S.Half)
    assert asech(-sqrt(2)) == I * acos(-1 / sqrt(2))
    assert asech(-2 / sqrt(3)) == I * acos(-sqrt(3) / 2)
    assert asech(-S(2)) == I * acos(Rational(-1, 2))
    assert asech(-2 / sqrt(2)) == I * acos(-sqrt(2) / 2)
    assert expand_mul(sech(asech(sqrt(6) - sqrt(2))) / (sqrt(6) - sqrt(2))) == 1
    assert expand_mul(sech(asech(sqrt(6) + sqrt(2))) / (sqrt(6) + sqrt(2))) == 1
    assert (sech(asech(sqrt(2 + 2 / sqrt(5)))) / sqrt(2 + 2 / sqrt(5))).simplify() == 1
    assert (sech(asech(-sqrt(2 + 2 / sqrt(5)))) / -sqrt(2 + 2 / sqrt(5))).simplify() == 1
    assert (sech(asech(sqrt(2 * (2 + sqrt(2))))) / sqrt(2 * (2 + sqrt(2)))).simplify() == 1
    assert expand_mul(sech(asech(1 + sqrt(5))) / (1 + sqrt(5))) == 1
    assert expand_mul(sech(asech(-1 - sqrt(5))) / (-1 - sqrt(5))) == 1
    assert expand_mul(sech(asech(-sqrt(6) - sqrt(2))) / (-sqrt(6) - sqrt(2))) == 1
    assert str(asech(5 * I).n(6)) == '0.19869 - 1.5708*I'
    assert str(asech(-5 * I).n(6)) == '0.19869 + 1.5708*I'