from sympy.core.function import Function
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan, sin, tan)
from sympy.solvers.ode import (classify_ode, checkinfsol, dsolve, infinitesimals)
from sympy.solvers.ode.subscheck import checkodesol
from sympy.testing.pytest import XFAIL
def test_user_infinitesimals():
    x = Symbol('x')
    eq = x * f(x).diff(x) + 1 - f(x) ** 2
    sol = Eq(f(x), (C1 + x ** 2) / (C1 - x ** 2))
    infinitesimals = {'xi': sqrt(f(x) - 1) / sqrt(f(x) + 1), 'eta': 0}
    assert dsolve(eq, hint='lie_group', **infinitesimals) == sol
    assert checkodesol(eq, sol) == (True, 0)