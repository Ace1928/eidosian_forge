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
def test_asech_nseries():
    x = Symbol('x')
    assert asech(x + 1)._eval_nseries(x, 4, None) == sqrt(2) * sqrt(-x) + 5 * sqrt(2) * (-x) ** (S(3) / 2) / 12 + 43 * sqrt(2) * (-x) ** (S(5) / 2) / 160 + 177 * sqrt(2) * (-x) ** (S(7) / 2) / 896 + O(x ** 4)
    assert asech(x - 1)._eval_nseries(x, 4, None) == I * pi + sqrt(2) * sqrt(x) + 5 * sqrt(2) * x ** (S(3) / 2) / 12 + 43 * sqrt(2) * x ** (S(5) / 2) / 160 + 177 * sqrt(2) * x ** (S(7) / 2) / 896 + O(x ** 4)
    assert asech(I * x + 3)._eval_nseries(x, 4, None) == -asech(3) + sqrt(2) * x / 12 - 17 * sqrt(2) * I * x ** 2 / 576 - 443 * sqrt(2) * x ** 3 / 41472 + O(x ** 4)
    assert asech(-I * x + 3)._eval_nseries(x, 4, None) == asech(3) + sqrt(2) * x / 12 + 17 * sqrt(2) * I * x ** 2 / 576 - 443 * sqrt(2) * x ** 3 / 41472 + O(x ** 4)
    assert asech(I * x - 3)._eval_nseries(x, 4, None) == -asech(-3) - sqrt(2) * x / 12 - 17 * sqrt(2) * I * x ** 2 / 576 + 443 * sqrt(2) * x ** 3 / 41472 + O(x ** 4)
    assert asech(-I * x - 3)._eval_nseries(x, 4, None) == asech(-3) - sqrt(2) * x / 12 + 17 * sqrt(2) * I * x ** 2 / 576 + 443 * sqrt(2) * x ** 3 / 41472 + O(x ** 4)
    assert asech(-I * x ** 2 + x - 2)._eval_nseries(x, 3, None) == 2 * I * pi / 3 + sqrt(3) * I * x / 6 + x ** 2 * (sqrt(3) / 6 + 7 * sqrt(3) * I / 72) + O(x ** 3)