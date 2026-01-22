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
def test_issue_3210():
    eqs = (sin(2) * cos(3) + sin(3) * cos(2), -sin(2) * sin(3) + cos(2) * cos(3), sin(2) * cos(3) - sin(3) * cos(2), sin(2) * sin(3) + cos(2) * cos(3), sin(2) * sin(3) + cos(2) * cos(3) + cos(2), sinh(2) * cosh(3) + sinh(3) * cosh(2), sinh(2) * sinh(3) + cosh(2) * cosh(3))
    assert [trigsimp(e) for e in eqs] == [sin(5), cos(5), -sin(1), cos(1), cos(1) + cos(2), sinh(5), cosh(5)]