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
def test_acoth():
    x = Symbol('x')
    assert acoth(0) == I * pi / 2
    assert acoth(I) == -I * pi / 4
    assert acoth(-I) == I * pi / 4
    assert acoth(1) is oo
    assert acoth(-1) is -oo
    assert acoth(nan) is nan
    assert acoth(oo) == 0
    assert acoth(-oo) == 0
    assert acoth(I * oo) == 0
    assert acoth(-I * oo) == 0
    assert acoth(zoo) == 0
    assert acoth(-x) == -acoth(x)
    assert acoth(I / sqrt(3)) == -I * pi / 3
    assert acoth(-I / sqrt(3)) == I * pi / 3
    assert acoth(I * sqrt(3)) == -I * pi / 6
    assert acoth(-I * sqrt(3)) == I * pi / 6
    assert acoth(I * (1 + sqrt(2))) == -pi * I / 8
    assert acoth(-I * (sqrt(2) + 1)) == pi * I / 8
    assert acoth(I * (1 - sqrt(2))) == pi * I * Rational(3, 8)
    assert acoth(I * (sqrt(2) - 1)) == pi * I * Rational(-3, 8)
    assert acoth(I * sqrt(5 + 2 * sqrt(5))) == -I * pi / 10
    assert acoth(-I * sqrt(5 + 2 * sqrt(5))) == I * pi / 10
    assert acoth(I * (2 + sqrt(3))) == -pi * I / 12
    assert acoth(-I * (2 + sqrt(3))) == pi * I / 12
    assert acoth(I * (2 - sqrt(3))) == pi * I * Rational(-5, 12)
    assert acoth(I * (sqrt(3) - 2)) == pi * I * Rational(5, 12)
    assert acoth(Rational(-1, 2)) == -acoth(S.Half)