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
def test_heuristic_linear():
    a, b, m, n = symbols('a b m n')
    eq = x ** (n * (m + 1) - m) * f(x).diff(x) - a * f(x) ** n - b * x ** (n * (m + 1))
    i = infinitesimals(eq, hint='linear')
    assert checkinfsol(eq, i)[0]