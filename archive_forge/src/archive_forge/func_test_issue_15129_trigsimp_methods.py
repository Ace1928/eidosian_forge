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
def test_issue_15129_trigsimp_methods():
    t1 = Matrix([sin(Rational(1, 50)), cos(Rational(1, 50)), 0])
    t2 = Matrix([sin(Rational(1, 25)), cos(Rational(1, 25)), 0])
    t3 = Matrix([cos(Rational(1, 25)), sin(Rational(1, 25)), 0])
    r1 = t1.dot(t2)
    r2 = t1.dot(t3)
    assert trigsimp(r1) == cos(Rational(1, 50))
    assert trigsimp(r2) == sin(Rational(3, 50))