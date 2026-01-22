from itertools import product
from sympy.core.function import (Subs, count_ops, diff, expand)
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (cosh, coth, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, cot, sin, tan)
from sympy.functions.elementary.trigonometric import (acos, asin, atan2)
from sympy.functions.elementary.trigonometric import (asec, acsc)
from sympy.functions.elementary.trigonometric import (acot, atan)
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import (exptrigsimp, trigsimp)
from sympy.testing.pytest import XFAIL
from sympy.abc import x, y
def test_trigsimp3():
    x, y = symbols('x,y')
    assert trigsimp(sin(x) / cos(x)) == tan(x)
    assert trigsimp(sin(x) ** 2 / cos(x) ** 2) == tan(x) ** 2
    assert trigsimp(sin(x) ** 3 / cos(x) ** 3) == tan(x) ** 3
    assert trigsimp(sin(x) ** 10 / cos(x) ** 10) == tan(x) ** 10
    assert trigsimp(cos(x) / sin(x)) == 1 / tan(x)
    assert trigsimp(cos(x) ** 2 / sin(x) ** 2) == 1 / tan(x) ** 2
    assert trigsimp(cos(x) ** 10 / sin(x) ** 10) == 1 / tan(x) ** 10
    assert trigsimp(tan(x)) == trigsimp(sin(x) / cos(x))