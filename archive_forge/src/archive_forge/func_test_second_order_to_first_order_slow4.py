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
def test_second_order_to_first_order_slow4():
    f, g = symbols('f g', cls=Function)
    x, t, x_, t_, d, a, m = symbols('x t x_ t_ d a m')
    eqs4 = [Eq(Derivative(f(t), (t, 2)), t * sin(t) * Derivative(g(t), t) - g(t) * sin(t)), Eq(Derivative(g(t), (t, 2)), t * sin(t) * Derivative(f(t), t) - f(t) * sin(t))]
    sol4 = [Eq(f(t), C1 * t + t * Integral(C2 * exp(-t_) * exp(exp(t_) * cos(exp(t_))) * exp(-sin(exp(t_))) / 2 + C2 * exp(-t_) * exp(-exp(t_) * cos(exp(t_))) * exp(sin(exp(t_))) / 2 - C3 * exp(-t_) * exp(exp(t_) * cos(exp(t_))) * exp(-sin(exp(t_))) / 2 + C3 * exp(-t_) * exp(-exp(t_) * cos(exp(t_))) * exp(sin(exp(t_))) / 2, (t_, log(t)))), Eq(g(t), C4 * t + t * Integral(-C2 * exp(-t_) * exp(exp(t_) * cos(exp(t_))) * exp(-sin(exp(t_))) / 2 + C2 * exp(-t_) * exp(-exp(t_) * cos(exp(t_))) * exp(sin(exp(t_))) / 2 + C3 * exp(-t_) * exp(exp(t_) * cos(exp(t_))) * exp(-sin(exp(t_))) / 2 + C3 * exp(-t_) * exp(-exp(t_) * cos(exp(t_))) * exp(sin(exp(t_))) / 2, (t_, log(t))))]
    assert dsolve_system(eqs4, simplify=False, doit=False) == [sol4]
    assert checksysodesol(eqs4, sol4) == (True, [0, 0])