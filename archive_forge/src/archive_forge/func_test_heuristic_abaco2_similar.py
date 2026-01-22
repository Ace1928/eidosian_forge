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
def test_heuristic_abaco2_similar():
    a, b = symbols('a b')
    F = Function('F')
    eq = f(x).diff(x) - F(a * x + b * f(x))
    i = infinitesimals(eq, hint='abaco2_similar')
    assert i == [{eta(x, f(x)): -a / b, xi(x, f(x)): 1}]
    assert checkinfsol(eq, i)[0]
    eq = f(x).diff(x) - f(x) ** 2 / (sin(f(x) - x) - x ** 2 + 2 * x * f(x))
    i = infinitesimals(eq, hint='abaco2_similar')
    assert i == [{eta(x, f(x)): f(x) ** 2, xi(x, f(x)): f(x) ** 2}]
    assert checkinfsol(eq, i)[0]