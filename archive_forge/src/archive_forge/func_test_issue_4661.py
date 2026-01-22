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
def test_issue_4661():
    a, x, y = symbols('a x y')
    eq = -4 * sin(x) ** 4 + 4 * cos(x) ** 4 - 8 * cos(x) ** 2
    assert trigsimp(eq) == -4
    n = sin(x) ** 6 + 4 * sin(x) ** 4 * cos(x) ** 2 + 5 * sin(x) ** 2 * cos(x) ** 4 + 2 * cos(x) ** 6
    d = -sin(x) ** 2 - 2 * cos(x) ** 2
    assert simplify(n / d) == -1
    assert trigsimp(-2 * cos(x) ** 2 + cos(x) ** 4 - sin(x) ** 4) == -1
    eq = -sin(x) ** 3 / 4 * cos(x) + cos(x) ** 3 / 4 * sin(x) - sin(2 * x) * cos(2 * x) / 8
    assert trigsimp(eq) == 0