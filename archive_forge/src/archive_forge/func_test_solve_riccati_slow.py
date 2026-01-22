from sympy.core.random import randint
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.polys.polytools import Poly
from sympy.simplify.ratsimp import ratsimp
from sympy.solvers.ode.subscheck import checkodesol
from sympy.testing.pytest import slow
from sympy.solvers.ode.riccati import (riccati_normal, riccati_inverse_normal,
@slow
def test_solve_riccati_slow():
    """
    This function tests the computation of rational
    particular solutions for a Riccati ODE.

    Each test case has 2 values -

    1. eq - Riccati ODE to be solved.
    2. sol - Expected solution to the equation.
    """
    C0 = Dummy('C0')
    tests = [(Eq(f(x).diff(x), (1 - x) * f(x) / (x - 3) + (2 - 12 * x) * f(x) ** 2 / (2 * x - 9) + (54924 * x ** 3 - 405264 * x ** 2 + 1084347 * x - 1087533) / (8 * x ** 4 - 132 * x ** 3 + 810 * x ** 2 - 2187 * x + 2187) + 495), [Eq(f(x), (18 * x + 6) / (2 * x - 9))])]
    for eq, sol in tests:
        check_dummy_sol(eq, sol, C0)