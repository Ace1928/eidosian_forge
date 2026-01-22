from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (asin, cos, sin, tan)
from sympy.polys.rationaltools import together
from sympy.series.limits import limit
def test_Limits_simple_3b():
    h = Symbol('h')
    assert limit(((x + h) ** 3 - x ** 3) / h, h, 0) == 3 * x ** 2
    assert limit(1 / (1 - x) - 3 / (1 - x ** 3), x, 1) == -1
    assert limit((sqrt(1 + x) - 1) / (root3(1 + x) - 1), x, 0) == Rational(3) / 2
    assert limit((sqrt(x) - 1) / (x - 1), x, 1) == Rational(1) / 2
    assert limit((sqrt(x) - 8) / (root3(x) - 4), x, 64) == 3
    assert limit((root3(x) - 1) / (root4(x) - 1), x, 1) == Rational(4) / 3
    assert limit((root3(x ** 2) - 2 * root3(x) + 1) / (x - 1) ** 2, x, 1) == Rational(1) / 9