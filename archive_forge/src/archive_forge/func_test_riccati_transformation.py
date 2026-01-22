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
def test_riccati_transformation():
    """
    This function tests the transformation of the
    solution of a Riccati ODE to the solution of
    its corresponding normal Riccati ODE.

    Each test case 4 values -

    1. w - The solution to be transformed
    2. b1 - The coefficient of f(x) in the ODE.
    3. b2 - The coefficient of f(x)**2 in the ODE.
    4. y - The solution to the normal Riccati ODE.
    """
    tests = [(x / (x - 1), (x ** 2 + 7) / 3 * x, x, -x ** 2 / (x - 1) - x * (x ** 2 / 3 + S(7) / 3) / 2 - 1 / (2 * x)), ((2 * x + 3) / (2 * x + 2), (3 - 3 * x) / (x + 1), 5 * x, -5 * x * (2 * x + 3) / (2 * x + 2) - (3 - 3 * x) / Mul(2, x + 1, evaluate=False) - 1 / (2 * x)), (-1 / (2 * x ** 2 - 1), 0, (2 - x) / (4 * x - 2), (2 - x) / ((4 * x - 2) * (2 * x ** 2 - 1)) - (4 * x - 2) * (Mul(-4, 2 - x, evaluate=False) / (4 * x - 2) ** 2 - 1 / (4 * x - 2)) / Mul(2, 2 - x, evaluate=False)), (x, (8 * x - 12) / (12 * x + 9), x ** 3 / (6 * x - 9), -x ** 4 / (6 * x - 9) - (8 * x - 12) / Mul(2, 12 * x + 9, evaluate=False) - (6 * x - 9) * (-6 * x ** 3 / (6 * x - 9) ** 2 + 3 * x ** 2 / (6 * x - 9)) / (2 * x ** 3))]
    for w, b1, b2, y in tests:
        assert y == riccati_normal(w, x, b1, b2)
        assert w == riccati_inverse_normal(y, x, b1, b2).cancel()
    tests = [((-2 * x - 1) / (2 * x ** 2 + 2 * x - 2), -2 / x, (-x - 1) / (4 * x), 8 * x ** 2 * (1 / (4 * x) + (-x - 1) / (4 * x ** 2)) / (-x - 1) ** 2 + 4 / (-x - 1), -2 * x * (-1 / (4 * x) - (-x - 1) / (4 * x ** 2)) / (-x - 1) - (-2 * x - 1) * (-x - 1) / (4 * x * (2 * x ** 2 + 2 * x - 2)) + 1 / x), (3 / (2 * x ** 2), -2 / x, (-x - 1) / (4 * x), 8 * x ** 2 * (1 / (4 * x) + (-x - 1) / (4 * x ** 2)) / (-x - 1) ** 2 + 4 / (-x - 1), -2 * x * (-1 / (4 * x) - (-x - 1) / (4 * x ** 2)) / (-x - 1) + 1 / x - Mul(3, -x - 1, evaluate=False) / (8 * x ** 3))]
    for w, b1, b2, bp, y in tests:
        assert y == riccati_normal(w, x, b1, b2)
        assert w == riccati_inverse_normal(y, x, b1, b2, bp).cancel()