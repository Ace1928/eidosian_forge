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
def test_sysode_linear_neq_order1_type5_type6():
    f, g = symbols('f g', cls=Function)
    x, x_ = symbols('x x_')
    eqs1 = [Eq(Derivative(f(x), x), (2 * f(x) + g(x)) / x), Eq(Derivative(g(x), x), (f(x) + 2 * g(x)) / x)]
    sol1 = [Eq(f(x), -C1 * x + C2 * x ** 3), Eq(g(x), C1 * x + C2 * x ** 3)]
    assert dsolve(eqs1) == sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0])
    eqs2 = [Eq(Derivative(f(x), x), (2 * f(x) + g(x) + 1) / x), Eq(Derivative(g(x), x), (x + f(x) + 2 * g(x)) / x)]
    sol2 = [Eq(f(x), C2 * x ** 3 - x * (C1 + Rational(1, 4)) + x * log(x) * Rational(-1, 2) + Rational(-2, 3)), Eq(g(x), C2 * x ** 3 + x * log(x) / 2 + x * (C1 + Rational(-1, 4)) + Rational(1, 3))]
    assert dsolve(eqs2) == sol2
    assert checksysodesol(eqs2, sol2) == (True, [0, 0])