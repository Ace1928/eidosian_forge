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
def test_trigsimp2():
    x, y = symbols('x,y')
    assert trigsimp(cos(x) ** 2 * sin(y) ** 2 + cos(x) ** 2 * cos(y) ** 2 + sin(x) ** 2, recursive=True) == 1
    assert trigsimp(sin(x) ** 2 * sin(y) ** 2 + sin(x) ** 2 * cos(y) ** 2 + cos(x) ** 2, recursive=True) == 1
    assert trigsimp(Subs(x, x, sin(y) ** 2 + cos(y) ** 2)) == Subs(x, x, 1)