from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.hyperbolic import sinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.core.containers import Tuple
from sympy.functions import exp, cos, sin, log, Ci, Si, erf, erfi
from sympy.matrices import dotprodsimp, NonSquareMatrixError
from sympy.solvers.ode import dsolve
from sympy.solvers.ode.ode import constant_renumber
from sympy.solvers.ode.subscheck import checksysodesol
from sympy.solvers.ode.systems import (_classify_linear_system, linear_ode_to_matrix,
from sympy.functions import airyai, airybi
from sympy.integrals.integrals import Integral
from sympy.simplify.ratsimp import ratsimp
from sympy.testing.pytest import ON_CI, raises, slow, skip, XFAIL
@slow
def test_higher_order_to_first_order_9():
    f, g = symbols('f g', cls=Function)
    x = symbols('x')
    eqs9 = [f(x) + g(x) - 2 * exp(I * x) + 2 * Derivative(f(x), x) + Derivative(f(x), (x, 2)), f(x) + g(x) - 2 * exp(I * x) + 2 * Derivative(g(x), x) + Derivative(g(x), (x, 2))]
    sol9 = [Eq(f(x), -C1 + C4 * exp(-2 * x) / 2 - (C2 / 2 - C3 / 2) * exp(-x) * cos(x) + (C2 / 2 + C3 / 2) * exp(-x) * sin(x) + 2 * ((1 - 2 * I) * exp(I * x) * sin(x) ** 2 / 5) + 2 * ((1 - 2 * I) * exp(I * x) * cos(x) ** 2 / 5)), Eq(g(x), C1 - C4 * exp(-2 * x) / 2 - (C2 / 2 - C3 / 2) * exp(-x) * cos(x) + (C2 / 2 + C3 / 2) * exp(-x) * sin(x) + 2 * ((1 - 2 * I) * exp(I * x) * sin(x) ** 2 / 5) + 2 * ((1 - 2 * I) * exp(I * x) * cos(x) ** 2 / 5))]
    assert dsolve(eqs9) == sol9
    assert checksysodesol(eqs9, sol9) == (True, [0, 0])