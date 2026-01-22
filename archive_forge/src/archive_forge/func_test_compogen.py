from sympy.solvers.decompogen import decompogen, compogen
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt, Max
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.testing.pytest import XFAIL, raises
def test_compogen():
    assert compogen([sin(x), cos(x)], x) == sin(cos(x))
    assert compogen([x ** 2 + x + 1, sin(x)], x) == sin(x) ** 2 + sin(x) + 1
    assert compogen([sqrt(x), 6 * x ** 2 - 5], x) == sqrt(6 * x ** 2 - 5)
    assert compogen([sin(x), sqrt(x), cos(x), x ** 2 + 1], x) == sin(sqrt(cos(x ** 2 + 1)))
    assert compogen([Abs(x), x ** 2 + 3 * x - 4, cos(x)], x) == Abs(cos(x) ** 2 + 3 * cos(x) - 4)
    assert compogen([x ** 2 + x - sqrt(3) / 2, sin(x)], x) == sin(x) ** 2 + sin(x) - sqrt(3) / 2
    assert compogen([Abs(x), 3 * x + cos(y) ** 2 - 4, cos(x)], x) == Abs(3 * cos(x) + cos(y) ** 2 - 4)
    assert compogen([x ** 2 + 2 * x + 1, x ** 2], x) == x ** 4 + 2 * x ** 2 + 1
    assert compogen([x ** 2 - x - 1, x ** 2 + x], x) == -x ** 2 - x + (x ** 2 + x) ** 2 - 1