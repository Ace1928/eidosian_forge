from sympy.concrete.summations import Sum
from sympy.core.function import Function
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, Or)
from sympy.polys.polytools import (Poly, PurePoly)
from sympy.sets.sets import (FiniteSet, Interval, Union)
from sympy.solvers.inequalities import (reduce_inequalities,
from sympy.polys.rootoftools import rootof
from sympy.solvers.solvers import solve
from sympy.solvers.solveset import solveset
from sympy.abc import x, y
from sympy.core.mod import Mod
from sympy.testing.pytest import raises, XFAIL
def test_integer_domain_relational_isolve():
    dom = FiniteSet(0, 3)
    x = Symbol('x', zero=False)
    assert isolve((x - 1) * (x - 2) * (x - 4) < 0, x, domain=dom) == Eq(x, 3)
    x = Symbol('x')
    assert isolve(x + 2 < 0, x, domain=S.Integers) == (x <= -3) & (x > -oo) & Eq(Mod(x, 1), 0)
    assert isolve(2 * x + 3 > 0, x, domain=S.Integers) == (x >= -1) & (x < oo) & Eq(Mod(x, 1), 0)
    assert isolve(x ** 2 + 3 * x - 2 < 0, x, domain=S.Integers) == (x >= -3) & (x <= 0) & Eq(Mod(x, 1), 0)
    assert isolve(x ** 2 + 3 * x - 2 > 0, x, domain=S.Integers) == (x >= 1) & (x < oo) & Eq(Mod(x, 1), 0) | (x <= -4) & (x > -oo) & Eq(Mod(x, 1), 0)