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
def test_acoth_leading_term():
    x = Symbol('x')
    assert acoth(x + 1).as_leading_term(x, cdir=1) == -log(x) / 2 + log(2) / 2
    assert acoth(x + 1).as_leading_term(x, cdir=-1) == -log(x) / 2 + log(2) / 2
    assert acoth(x - 1).as_leading_term(x, cdir=1) == log(x) / 2 - log(2) / 2 + I * pi / 2
    assert acoth(x - 1).as_leading_term(x, cdir=-1) == log(x) / 2 - log(2) / 2 - I * pi / 2
    assert acoth(x).as_leading_term(x, cdir=-1) == I * pi / 2
    assert acoth(x).as_leading_term(x, cdir=1) == -I * pi / 2
    assert acoth(I * x + 1 / 2).as_leading_term(x, cdir=1) == acoth(1 / 2)
    assert acoth(-I * x + 1 / 2).as_leading_term(x, cdir=1) == acoth(1 / 2) + I * pi
    assert acoth(I * x - 1 / 2).as_leading_term(x, cdir=1) == -I * pi - acoth(1 / 2)
    assert acoth(-I * x - 1 / 2).as_leading_term(x, cdir=1) == -acoth(1 / 2)
    assert acoth(-I * x ** 2 - x - S(1) / 2).as_leading_term(x, cdir=1) == -log(3) / 2 + I * pi / 2
    assert acoth(-I * x ** 2 - x - S(1) / 2).as_leading_term(x, cdir=-1) == -log(3) / 2 + I * pi / 2