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
def test_canonical_odes():
    f, g, h = symbols('f g h', cls=Function)
    x = symbols('x')
    funcs = [f(x), g(x), h(x)]
    eqs1 = [Eq(f(x).diff(x, x), f(x) + 2 * g(x)), Eq(g(x) + 1, g(x).diff(x) + f(x))]
    sol1 = [[Eq(Derivative(f(x), (x, 2)), f(x) + 2 * g(x)), Eq(Derivative(g(x), x), -f(x) + g(x) + 1)]]
    assert canonical_odes(eqs1, funcs[:2], x) == sol1
    eqs2 = [Eq(f(x).diff(x), h(x).diff(x) + f(x)), Eq(g(x).diff(x) ** 2, f(x) + h(x)), Eq(h(x).diff(x), f(x))]
    sol2 = [[Eq(Derivative(f(x), x), 2 * f(x)), Eq(Derivative(g(x), x), -sqrt(f(x) + h(x))), Eq(Derivative(h(x), x), f(x))], [Eq(Derivative(f(x), x), 2 * f(x)), Eq(Derivative(g(x), x), sqrt(f(x) + h(x))), Eq(Derivative(h(x), x), f(x))]]
    assert canonical_odes(eqs2, funcs, x) == sol2