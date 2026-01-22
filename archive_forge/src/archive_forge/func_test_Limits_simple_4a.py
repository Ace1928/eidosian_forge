from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (asin, cos, sin, tan)
from sympy.polys.rationaltools import together
from sympy.series.limits import limit
def test_Limits_simple_4a():
    a = Symbol('a')
    assert limit((sqrt(x) - sqrt(a)) / (x - a), x, a) == 1 / (2 * sqrt(a))
    assert limit((sqrt(x) - 1) / (root3(x) - 1), x, 1) == Rational(3, 2)
    assert limit((sqrt(1 + x) - sqrt(1 - x)) / x, x, 0) == 1
    assert limit(sqrt(x ** 2 - 5 * x + 6) - x, x, oo) == Rational(-5, 2)