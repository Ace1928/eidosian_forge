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
def test_second_order_to_first_order_2():
    f, g = symbols('f g', cls=Function)
    x, t, x_, t_, d, a, m = symbols('x t x_ t_ d a m')
    eqs2 = [Eq(f(x).diff(x, 2), 2 * (x * g(x).diff(x) - g(x))), Eq(g(x).diff(x, 2), -2 * (x * f(x).diff(x) - f(x)))]
    sol2 = [Eq(f(x), C1 * x + x * Integral(C2 * exp(-x_) * exp(I * exp(2 * x_)) / 2 + C2 * exp(-x_) * exp(-I * exp(2 * x_)) / 2 - I * C3 * exp(-x_) * exp(I * exp(2 * x_)) / 2 + I * C3 * exp(-x_) * exp(-I * exp(2 * x_)) / 2, (x_, log(x)))), Eq(g(x), C4 * x + x * Integral(I * C2 * exp(-x_) * exp(I * exp(2 * x_)) / 2 - I * C2 * exp(-x_) * exp(-I * exp(2 * x_)) / 2 + C3 * exp(-x_) * exp(I * exp(2 * x_)) / 2 + C3 * exp(-x_) * exp(-I * exp(2 * x_)) / 2, (x_, log(x))))]
    assert dsolve_system(eqs2, simplify=False, doit=False) == [sol2]
    assert checksysodesol(eqs2, sol2) == (True, [0, 0])
    eqs3 = (Eq(diff(f(t), t, t), 9 * t * diff(g(t), t) - 9 * g(t)), Eq(diff(g(t), t, t), 7 * t * diff(f(t), t) - 7 * f(t)))
    sol3 = [Eq(f(t), C1 * t + t * Integral(C2 * exp(-t_) * exp(3 * sqrt(7) * exp(2 * t_) / 2) / 2 + C2 * exp(-t_) * exp(-3 * sqrt(7) * exp(2 * t_) / 2) / 2 + 3 * sqrt(7) * C3 * exp(-t_) * exp(3 * sqrt(7) * exp(2 * t_) / 2) / 14 - 3 * sqrt(7) * C3 * exp(-t_) * exp(-3 * sqrt(7) * exp(2 * t_) / 2) / 14, (t_, log(t)))), Eq(g(t), C4 * t + t * Integral(sqrt(7) * C2 * exp(-t_) * exp(3 * sqrt(7) * exp(2 * t_) / 2) / 6 - sqrt(7) * C2 * exp(-t_) * exp(-3 * sqrt(7) * exp(2 * t_) / 2) / 6 + C3 * exp(-t_) * exp(3 * sqrt(7) * exp(2 * t_) / 2) / 2 + C3 * exp(-t_) * exp(-3 * sqrt(7) * exp(2 * t_) / 2) / 2, (t_, log(t))))]
    assert dsolve_system(eqs3, simplify=False, doit=False) == [sol3]
    assert checksysodesol(eqs3, sol3) == (True, [0, 0])
    eqs5 = [Eq(Derivative(g(t), (t, 2)), a * m), Eq(Derivative(f(t), (t, 2)), 0)]
    sol5 = [Eq(g(t), C1 + C2 * t + a * m * t ** 2 / 2), Eq(f(t), C3 + C4 * t)]
    assert dsolve(eqs5) == sol5
    assert checksysodesol(eqs5, sol5) == (True, [0, 0])
    eqs6 = [Eq(Derivative(f(t), (t, 2)), f(t) / t ** 4), Eq(Derivative(g(t), (t, 2)), d * g(t) / t ** 4)]
    sol6 = [Eq(f(t), C1 * sqrt(t ** 2) * exp(-1 / t) - C2 * sqrt(t ** 2) * exp(1 / t)), Eq(g(t), C3 * sqrt(t ** 2) * exp(-sqrt(d) / t) * d ** Rational(-1, 2) - C4 * sqrt(t ** 2) * exp(sqrt(d) / t) * d ** Rational(-1, 2))]
    assert dsolve(eqs6) == sol6
    assert checksysodesol(eqs6, sol6) == (True, [0, 0])