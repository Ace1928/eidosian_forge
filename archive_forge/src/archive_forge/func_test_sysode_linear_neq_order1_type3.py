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
def test_sysode_linear_neq_order1_type3():
    f, g, h, k, x0, y0 = symbols('f g h k x0 y0', cls=Function)
    x, t, a = symbols('x t a')
    r = symbols('r', real=True)
    eqs1 = [Eq(Derivative(f(r), r), r * g(r) + f(r)), Eq(Derivative(g(r), r), -r * f(r) + g(r))]
    sol1 = [Eq(f(r), C1 * exp(r) * sin(r ** 2 / 2) + C2 * exp(r) * cos(r ** 2 / 2)), Eq(g(r), C1 * exp(r) * cos(r ** 2 / 2) - C2 * exp(r) * sin(r ** 2 / 2))]
    assert dsolve(eqs1) == sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0])
    eqs2 = [Eq(Derivative(f(x), x), x ** 2 * g(x) + x * f(x)), Eq(Derivative(g(x), x), 2 * x ** 2 * f(x) + (3 * x ** 2 + x) * g(x))]
    sol2 = [Eq(f(x), (sqrt(17) * C1 / 17 + C2 * (17 - 3 * sqrt(17)) / 34) * exp(x ** 3 * (3 + sqrt(17)) / 6 + x ** 2 / 2) - exp(x ** 3 * (3 - sqrt(17)) / 6 + x ** 2 / 2) * (sqrt(17) * C1 / 17 + C2 * (3 * sqrt(17) + 17) * Rational(-1, 34))), Eq(g(x), exp(x ** 3 * (3 - sqrt(17)) / 6 + x ** 2 / 2) * (C1 * (17 - 3 * sqrt(17)) / 34 + sqrt(17) * C2 * Rational(-2, 17)) + exp(x ** 3 * (3 + sqrt(17)) / 6 + x ** 2 / 2) * (C1 * (3 * sqrt(17) + 17) / 34 + sqrt(17) * C2 * Rational(2, 17)))]
    assert dsolve(eqs2) == sol2
    assert checksysodesol(eqs2, sol2) == (True, [0, 0])
    eqs3 = [Eq(f(x).diff(x), x * f(x) + g(x)), Eq(g(x).diff(x), -f(x) + x * g(x))]
    sol3 = [Eq(f(x), (C1 / 2 + I * C2 / 2) * exp(x ** 2 / 2 - I * x) + exp(x ** 2 / 2 + I * x) * (C1 / 2 + I * C2 * Rational(-1, 2))), Eq(g(x), (I * C1 / 2 + C2 / 2) * exp(x ** 2 / 2 + I * x) - exp(x ** 2 / 2 - I * x) * (I * C1 / 2 + C2 * Rational(-1, 2)))]
    assert dsolve(eqs3) == sol3
    assert checksysodesol(eqs3, sol3) == (True, [0, 0])
    eqs4 = [Eq(f(x).diff(x), x * (f(x) + g(x) + h(x))), Eq(g(x).diff(x), x * (f(x) + g(x) + h(x))), Eq(h(x).diff(x), x * (f(x) + g(x) + h(x)))]
    sol4 = [Eq(f(x), -C1 / 3 - C2 / 3 + 2 * C3 / 3 + (C1 / 3 + C2 / 3 + C3 / 3) * exp(3 * x ** 2 / 2)), Eq(g(x), 2 * C1 / 3 - C2 / 3 - C3 / 3 + (C1 / 3 + C2 / 3 + C3 / 3) * exp(3 * x ** 2 / 2)), Eq(h(x), -C1 / 3 + 2 * C2 / 3 - C3 / 3 + (C1 / 3 + C2 / 3 + C3 / 3) * exp(3 * x ** 2 / 2))]
    assert dsolve(eqs4) == sol4
    assert checksysodesol(eqs4, sol4) == (True, [0, 0, 0])
    eqs5 = [Eq(f(x).diff(x), x ** 2 * (f(x) + g(x) + h(x))), Eq(g(x).diff(x), x ** 2 * (f(x) + g(x) + h(x))), Eq(h(x).diff(x), x ** 2 * (f(x) + g(x) + h(x)))]
    sol5 = [Eq(f(x), -C1 / 3 - C2 / 3 + 2 * C3 / 3 + (C1 / 3 + C2 / 3 + C3 / 3) * exp(x ** 3)), Eq(g(x), 2 * C1 / 3 - C2 / 3 - C3 / 3 + (C1 / 3 + C2 / 3 + C3 / 3) * exp(x ** 3)), Eq(h(x), -C1 / 3 + 2 * C2 / 3 - C3 / 3 + (C1 / 3 + C2 / 3 + C3 / 3) * exp(x ** 3))]
    assert dsolve(eqs5) == sol5
    assert checksysodesol(eqs5, sol5) == (True, [0, 0, 0])
    eqs6 = [Eq(Derivative(f(x), x), x * (f(x) + g(x) + h(x) + k(x))), Eq(Derivative(g(x), x), x * (f(x) + g(x) + h(x) + k(x))), Eq(Derivative(h(x), x), x * (f(x) + g(x) + h(x) + k(x))), Eq(Derivative(k(x), x), x * (f(x) + g(x) + h(x) + k(x)))]
    sol6 = [Eq(f(x), -C1 / 4 - C2 / 4 - C3 / 4 + 3 * C4 / 4 + (C1 / 4 + C2 / 4 + C3 / 4 + C4 / 4) * exp(2 * x ** 2)), Eq(g(x), 3 * C1 / 4 - C2 / 4 - C3 / 4 - C4 / 4 + (C1 / 4 + C2 / 4 + C3 / 4 + C4 / 4) * exp(2 * x ** 2)), Eq(h(x), -C1 / 4 + 3 * C2 / 4 - C3 / 4 - C4 / 4 + (C1 / 4 + C2 / 4 + C3 / 4 + C4 / 4) * exp(2 * x ** 2)), Eq(k(x), -C1 / 4 - C2 / 4 + 3 * C3 / 4 - C4 / 4 + (C1 / 4 + C2 / 4 + C3 / 4 + C4 / 4) * exp(2 * x ** 2))]
    assert dsolve(eqs6) == sol6
    assert checksysodesol(eqs6, sol6) == (True, [0, 0, 0, 0])
    y = symbols('y', real=True)
    eqs7 = [Eq(Derivative(f(y), y), y * f(y) + g(y)), Eq(Derivative(g(y), y), y * g(y) - f(y))]
    sol7 = [Eq(f(y), C1 * exp(y ** 2 / 2) * sin(y) + C2 * exp(y ** 2 / 2) * cos(y)), Eq(g(y), C1 * exp(y ** 2 / 2) * cos(y) - C2 * exp(y ** 2 / 2) * sin(y))]
    assert dsolve(eqs7) == sol7
    assert checksysodesol(eqs7, sol7) == (True, [0, 0])
    eqs8 = [Eq(Derivative(f(t), t), 5 * t * f(t) + 2 * h(t)), Eq(Derivative(h(t), t), 2 * f(t) + 5 * t * h(t))]
    sol8 = [Eq(f(t), Mul(-1, C1 / 2 - C2 / 2, evaluate=False) * exp(5 * t ** 2 / 2 - 2 * t) + (C1 / 2 + C2 / 2) * exp(5 * t ** 2 / 2 + 2 * t)), Eq(h(t), (C1 / 2 - C2 / 2) * exp(5 * t ** 2 / 2 - 2 * t) + (C1 / 2 + C2 / 2) * exp(5 * t ** 2 / 2 + 2 * t))]
    assert dsolve(eqs8) == sol8
    assert checksysodesol(eqs8, sol8) == (True, [0, 0])
    eqs9 = [Eq(diff(f(t), t), 5 * t * f(t) + t ** 2 * g(t)), Eq(diff(g(t), t), -t ** 2 * f(t) + 5 * t * g(t))]
    sol9 = [Eq(f(t), (C1 / 2 - I * C2 / 2) * exp(I * t ** 3 / 3 + 5 * t ** 2 / 2) + (C1 / 2 + I * C2 / 2) * exp(-I * t ** 3 / 3 + 5 * t ** 2 / 2)), Eq(g(t), Mul(-1, I * C1 / 2 - C2 / 2, evaluate=False) * exp(-I * t ** 3 / 3 + 5 * t ** 2 / 2) + (I * C1 / 2 + C2 / 2) * exp(I * t ** 3 / 3 + 5 * t ** 2 / 2))]
    assert dsolve(eqs9) == sol9
    assert checksysodesol(eqs9, sol9) == (True, [0, 0])
    eqs10 = [Eq(diff(f(t), t), t ** 2 * g(t) + 5 * t * f(t)), Eq(diff(g(t), t), -t ** 2 * f(t) + (9 * t ** 2 + 5 * t) * g(t))]
    sol10 = [Eq(f(t), (C1 * (77 - 9 * sqrt(77)) / 154 + sqrt(77) * C2 / 77) * exp(t ** 3 * (sqrt(77) + 9) / 6 + 5 * t ** 2 / 2) + (C1 * (77 + 9 * sqrt(77)) / 154 - sqrt(77) * C2 / 77) * exp(t ** 3 * (9 - sqrt(77)) / 6 + 5 * t ** 2 / 2)), Eq(g(t), (sqrt(77) * C1 / 77 + C2 * (77 - 9 * sqrt(77)) / 154) * exp(t ** 3 * (9 - sqrt(77)) / 6 + 5 * t ** 2 / 2) - (sqrt(77) * C1 / 77 - C2 * (77 + 9 * sqrt(77)) / 154) * exp(t ** 3 * (sqrt(77) + 9) / 6 + 5 * t ** 2 / 2))]
    assert dsolve(eqs10) == sol10
    assert checksysodesol(eqs10, sol10) == (True, [0, 0])
    eqs11 = [Eq(diff(f(t), t), 5 * t * f(t) + t ** 2 * g(t)), Eq(diff(g(t), t), (1 - t ** 2) * f(t) + (5 * t + 9 * t ** 2) * g(t))]
    sol11 = [Eq(f(t), C1 * x0(t) + C2 * x0(t) * Integral(t ** 2 * exp(Integral(5 * t, t)) * exp(Integral(9 * t ** 2 + 5 * t, t)) / x0(t) ** 2, t)), Eq(g(t), C1 * y0(t) + C2 * (y0(t) * Integral(t ** 2 * exp(Integral(5 * t, t)) * exp(Integral(9 * t ** 2 + 5 * t, t)) / x0(t) ** 2, t) + exp(Integral(5 * t, t)) * exp(Integral(9 * t ** 2 + 5 * t, t)) / x0(t)))]
    assert dsolve(eqs11) == sol11