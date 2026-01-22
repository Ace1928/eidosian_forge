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
def test_sysode_linear_neq_order1_type4():
    f, g, h, k = symbols('f g h k', cls=Function)
    x, t, a = symbols('x t a')
    r = symbols('r', real=True)
    eqs1 = [Eq(diff(f(r), r), f(r) + r * g(r) + r ** 2), Eq(diff(g(r), r), -r * f(r) + g(r) + r)]
    sol1 = [Eq(f(r), C1 * exp(r) * sin(r ** 2 / 2) + C2 * exp(r) * cos(r ** 2 / 2) + exp(r) * sin(r ** 2 / 2) * Integral(r ** 2 * exp(-r) * sin(r ** 2 / 2) + r * exp(-r) * cos(r ** 2 / 2), r) + exp(r) * cos(r ** 2 / 2) * Integral(r ** 2 * exp(-r) * cos(r ** 2 / 2) - r * exp(-r) * sin(r ** 2 / 2), r)), Eq(g(r), C1 * exp(r) * cos(r ** 2 / 2) - C2 * exp(r) * sin(r ** 2 / 2) - exp(r) * sin(r ** 2 / 2) * Integral(r ** 2 * exp(-r) * cos(r ** 2 / 2) - r * exp(-r) * sin(r ** 2 / 2), r) + exp(r) * cos(r ** 2 / 2) * Integral(r ** 2 * exp(-r) * sin(r ** 2 / 2) + r * exp(-r) * cos(r ** 2 / 2), r))]
    assert dsolve(eqs1) == sol1
    assert checksysodesol(eqs1, sol1) == (True, [0, 0])
    eqs2 = [Eq(diff(f(r), r), f(r) + r * g(r) + r), Eq(diff(g(r), r), -r * f(r) + g(r) + log(r))]
    sol2 = [Eq(f(r), C1 * exp(r) * sin(r ** 2 / 2) + C2 * exp(r) * cos(r ** 2 / 2) + exp(r) * sin(r ** 2 / 2) * Integral(r * exp(-r) * sin(r ** 2 / 2) + exp(-r) * log(r) * cos(r ** 2 / 2), r) + exp(r) * cos(r ** 2 / 2) * Integral(r * exp(-r) * cos(r ** 2 / 2) - exp(-r) * log(r) * sin(r ** 2 / 2), r)), Eq(g(r), C1 * exp(r) * cos(r ** 2 / 2) - C2 * exp(r) * sin(r ** 2 / 2) - exp(r) * sin(r ** 2 / 2) * Integral(r * exp(-r) * cos(r ** 2 / 2) - exp(-r) * log(r) * sin(r ** 2 / 2), r) + exp(r) * cos(r ** 2 / 2) * Integral(r * exp(-r) * sin(r ** 2 / 2) + exp(-r) * log(r) * cos(r ** 2 / 2), r))]
    assert dsolve_system(eqs2, simplify=False, doit=False) == [sol2]
    assert checksysodesol(eqs2, sol2) == (True, [0, 0])
    eqs3 = [Eq(Derivative(f(x), x), x * (f(x) + g(x) + h(x)) + x), Eq(Derivative(g(x), x), x * (f(x) + g(x) + h(x)) + x), Eq(Derivative(h(x), x), x * (f(x) + g(x) + h(x)) + 1)]
    sol3 = [Eq(f(x), C1 * Rational(-1, 3) + C2 * Rational(-1, 3) + C3 * Rational(2, 3) + x ** 2 / 6 + x * Rational(-1, 3) + (C1 / 3 + C2 / 3 + C3 / 3) * exp(x ** 2 * Rational(3, 2)) + sqrt(6) * sqrt(pi) * erf(sqrt(6) * x / 2) * exp(x ** 2 * Rational(3, 2)) / 18 + Rational(-2, 9)), Eq(g(x), C1 * Rational(2, 3) + C2 * Rational(-1, 3) + C3 * Rational(-1, 3) + x ** 2 / 6 + x * Rational(-1, 3) + (C1 / 3 + C2 / 3 + C3 / 3) * exp(x ** 2 * Rational(3, 2)) + sqrt(6) * sqrt(pi) * erf(sqrt(6) * x / 2) * exp(x ** 2 * Rational(3, 2)) / 18 + Rational(-2, 9)), Eq(h(x), C1 * Rational(-1, 3) + C2 * Rational(2, 3) + C3 * Rational(-1, 3) + x ** 2 * Rational(-1, 3) + x * Rational(2, 3) + (C1 / 3 + C2 / 3 + C3 / 3) * exp(x ** 2 * Rational(3, 2)) + sqrt(6) * sqrt(pi) * erf(sqrt(6) * x / 2) * exp(x ** 2 * Rational(3, 2)) / 18 + Rational(-2, 9))]
    assert dsolve(eqs3) == sol3
    assert checksysodesol(eqs3, sol3) == (True, [0, 0, 0])
    eqs4 = [Eq(Derivative(f(x), x), x * (f(x) + g(x) + h(x)) + sin(x)), Eq(Derivative(g(x), x), x * (f(x) + g(x) + h(x)) + sin(x)), Eq(Derivative(h(x), x), x * (f(x) + g(x) + h(x)) + sin(x))]
    sol4 = [Eq(f(x), C1 * Rational(-1, 3) + C2 * Rational(-1, 3) + C3 * Rational(2, 3) + (C1 / 3 + C2 / 3 + C3 / 3) * exp(x ** 2 * Rational(3, 2)) + Integral(sin(x) * exp(x ** 2 * Rational(-3, 2)), x) * exp(x ** 2 * Rational(3, 2))), Eq(g(x), C1 * Rational(2, 3) + C2 * Rational(-1, 3) + C3 * Rational(-1, 3) + (C1 / 3 + C2 / 3 + C3 / 3) * exp(x ** 2 * Rational(3, 2)) + Integral(sin(x) * exp(x ** 2 * Rational(-3, 2)), x) * exp(x ** 2 * Rational(3, 2))), Eq(h(x), C1 * Rational(-1, 3) + C2 * Rational(2, 3) + C3 * Rational(-1, 3) + (C1 / 3 + C2 / 3 + C3 / 3) * exp(x ** 2 * Rational(3, 2)) + Integral(sin(x) * exp(x ** 2 * Rational(-3, 2)), x) * exp(x ** 2 * Rational(3, 2)))]
    assert dsolve(eqs4) == sol4
    assert checksysodesol(eqs4, sol4) == (True, [0, 0, 0])
    eqs5 = [Eq(Derivative(f(x), x), x * (f(x) + g(x) + h(x) + k(x) + 1)), Eq(Derivative(g(x), x), x * (f(x) + g(x) + h(x) + k(x) + 1)), Eq(Derivative(h(x), x), x * (f(x) + g(x) + h(x) + k(x) + 1)), Eq(Derivative(k(x), x), x * (f(x) + g(x) + h(x) + k(x) + 1))]
    sol5 = [Eq(f(x), C1 * Rational(-1, 4) + C2 * Rational(-1, 4) + C3 * Rational(-1, 4) + C4 * Rational(3, 4) + (C1 / 4 + C2 / 4 + C3 / 4 + C4 / 4) * exp(2 * x ** 2) + Rational(-1, 4)), Eq(g(x), C1 * Rational(3, 4) + C2 * Rational(-1, 4) + C3 * Rational(-1, 4) + C4 * Rational(-1, 4) + (C1 / 4 + C2 / 4 + C3 / 4 + C4 / 4) * exp(2 * x ** 2) + Rational(-1, 4)), Eq(h(x), C1 * Rational(-1, 4) + C2 * Rational(3, 4) + C3 * Rational(-1, 4) + C4 * Rational(-1, 4) + (C1 / 4 + C2 / 4 + C3 / 4 + C4 / 4) * exp(2 * x ** 2) + Rational(-1, 4)), Eq(k(x), C1 * Rational(-1, 4) + C2 * Rational(-1, 4) + C3 * Rational(3, 4) + C4 * Rational(-1, 4) + (C1 / 4 + C2 / 4 + C3 / 4 + C4 / 4) * exp(2 * x ** 2) + Rational(-1, 4))]
    assert dsolve(eqs5) == sol5
    assert checksysodesol(eqs5, sol5) == (True, [0, 0, 0, 0])
    eqs6 = [Eq(Derivative(f(x), x), x ** 2 * (f(x) + g(x) + h(x) + k(x) + 1)), Eq(Derivative(g(x), x), x ** 2 * (f(x) + g(x) + h(x) + k(x) + 1)), Eq(Derivative(h(x), x), x ** 2 * (f(x) + g(x) + h(x) + k(x) + 1)), Eq(Derivative(k(x), x), x ** 2 * (f(x) + g(x) + h(x) + k(x) + 1))]
    sol6 = [Eq(f(x), C1 * Rational(-1, 4) + C2 * Rational(-1, 4) + C3 * Rational(-1, 4) + C4 * Rational(3, 4) + (C1 / 4 + C2 / 4 + C3 / 4 + C4 / 4) * exp(x ** 3 * Rational(4, 3)) + Rational(-1, 4)), Eq(g(x), C1 * Rational(3, 4) + C2 * Rational(-1, 4) + C3 * Rational(-1, 4) + C4 * Rational(-1, 4) + (C1 / 4 + C2 / 4 + C3 / 4 + C4 / 4) * exp(x ** 3 * Rational(4, 3)) + Rational(-1, 4)), Eq(h(x), C1 * Rational(-1, 4) + C2 * Rational(3, 4) + C3 * Rational(-1, 4) + C4 * Rational(-1, 4) + (C1 / 4 + C2 / 4 + C3 / 4 + C4 / 4) * exp(x ** 3 * Rational(4, 3)) + Rational(-1, 4)), Eq(k(x), C1 * Rational(-1, 4) + C2 * Rational(-1, 4) + C3 * Rational(3, 4) + C4 * Rational(-1, 4) + (C1 / 4 + C2 / 4 + C3 / 4 + C4 / 4) * exp(x ** 3 * Rational(4, 3)) + Rational(-1, 4))]
    assert dsolve(eqs6) == sol6
    assert checksysodesol(eqs6, sol6) == (True, [0, 0, 0, 0])
    eqs7 = [Eq(Derivative(f(x), x), (f(x) + g(x) + h(x)) * log(x) + sin(x)), Eq(Derivative(g(x), x), (f(x) + g(x) + h(x)) * log(x) + sin(x)), Eq(Derivative(h(x), x), (f(x) + g(x) + h(x)) * log(x) + sin(x))]
    sol7 = [Eq(f(x), -C1 / 3 - C2 / 3 + 2 * C3 / 3 + (C1 / 3 + C2 / 3 + C3 / 3) * exp(x * (3 * log(x) - 3)) + exp(x * (3 * log(x) - 3)) * Integral(exp(3 * x) * exp(-3 * x * log(x)) * sin(x), x)), Eq(g(x), 2 * C1 / 3 - C2 / 3 - C3 / 3 + (C1 / 3 + C2 / 3 + C3 / 3) * exp(x * (3 * log(x) - 3)) + exp(x * (3 * log(x) - 3)) * Integral(exp(3 * x) * exp(-3 * x * log(x)) * sin(x), x)), Eq(h(x), -C1 / 3 + 2 * C2 / 3 - C3 / 3 + (C1 / 3 + C2 / 3 + C3 / 3) * exp(x * (3 * log(x) - 3)) + exp(x * (3 * log(x) - 3)) * Integral(exp(3 * x) * exp(-3 * x * log(x)) * sin(x), x))]
    with dotprodsimp(True):
        assert dsolve(eqs7, simplify=False, doit=False) == sol7
    assert checksysodesol(eqs7, sol7) == (True, [0, 0, 0])
    eqs8 = [Eq(Derivative(f(x), x), (f(x) + g(x) + h(x) + k(x)) * log(x) + sin(x)), Eq(Derivative(g(x), x), (f(x) + g(x) + h(x) + k(x)) * log(x) + sin(x)), Eq(Derivative(h(x), x), (f(x) + g(x) + h(x) + k(x)) * log(x) + sin(x)), Eq(Derivative(k(x), x), (f(x) + g(x) + h(x) + k(x)) * log(x) + sin(x))]
    sol8 = [Eq(f(x), -C1 / 4 - C2 / 4 - C3 / 4 + 3 * C4 / 4 + (C1 / 4 + C2 / 4 + C3 / 4 + C4 / 4) * exp(x * (4 * log(x) - 4)) + exp(x * (4 * log(x) - 4)) * Integral(exp(4 * x) * exp(-4 * x * log(x)) * sin(x), x)), Eq(g(x), 3 * C1 / 4 - C2 / 4 - C3 / 4 - C4 / 4 + (C1 / 4 + C2 / 4 + C3 / 4 + C4 / 4) * exp(x * (4 * log(x) - 4)) + exp(x * (4 * log(x) - 4)) * Integral(exp(4 * x) * exp(-4 * x * log(x)) * sin(x), x)), Eq(h(x), -C1 / 4 + 3 * C2 / 4 - C3 / 4 - C4 / 4 + (C1 / 4 + C2 / 4 + C3 / 4 + C4 / 4) * exp(x * (4 * log(x) - 4)) + exp(x * (4 * log(x) - 4)) * Integral(exp(4 * x) * exp(-4 * x * log(x)) * sin(x), x)), Eq(k(x), -C1 / 4 - C2 / 4 + 3 * C3 / 4 - C4 / 4 + (C1 / 4 + C2 / 4 + C3 / 4 + C4 / 4) * exp(x * (4 * log(x) - 4)) + exp(x * (4 * log(x) - 4)) * Integral(exp(4 * x) * exp(-4 * x * log(x)) * sin(x), x))]
    with dotprodsimp(True):
        assert dsolve(eqs8) == sol8
    assert checksysodesol(eqs8, sol8) == (True, [0, 0, 0, 0])