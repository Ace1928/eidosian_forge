from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import (ceiling, floor, frac)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, cos, tan
from sympy.core.expr import unchanged
from sympy.testing.pytest import XFAIL
def test_issue_18689():
    assert floor(floor(floor(x)) + 3) == floor(x) + 3
    assert ceiling(ceiling(ceiling(x)) + 1) == ceiling(x) + 1
    assert ceiling(ceiling(floor(x)) + 3) == floor(x) + 3