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
def test_dsolve_system():
    f, g = symbols('f g', cls=Function)
    x = symbols('x')
    eqs = [Eq(f(x).diff(x), f(x) + g(x)), Eq(g(x).diff(x), f(x) + g(x))]
    funcs = [f(x), g(x)]
    sol = [[Eq(f(x), -C1 + C2 * exp(2 * x)), Eq(g(x), C1 + C2 * exp(2 * x))]]
    assert dsolve_system(eqs, funcs=funcs, t=x, doit=True) == sol
    raises(ValueError, lambda: dsolve_system(1))
    raises(ValueError, lambda: dsolve_system(eqs, 1))
    raises(ValueError, lambda: dsolve_system(eqs, funcs, 1))
    raises(ValueError, lambda: dsolve_system(eqs, funcs[:1], x))
    eq = (Eq(f(x).diff(x), 12 * f(x) - 6 * g(x)), Eq(g(x).diff(x) ** 2, 11 * f(x) + 3 * g(x)))
    raises(NotImplementedError, lambda: dsolve_system(eq) == ([], []))
    raises(NotImplementedError, lambda: dsolve_system(eq, funcs=[f(x), g(x)]) == ([], []))
    raises(NotImplementedError, lambda: dsolve_system(eq, funcs=[f(x), g(x)], t=x) == ([], []))
    raises(NotImplementedError, lambda: dsolve_system(eq, funcs=[f(x), g(x)], t=x, ics={f(0): 1, g(0): 1}) == ([], []))
    raises(NotImplementedError, lambda: dsolve_system(eq, t=x, ics={f(0): 1, g(0): 1}) == ([], []))
    raises(NotImplementedError, lambda: dsolve_system(eq, ics={f(0): 1, g(0): 1}) == ([], []))
    raises(NotImplementedError, lambda: dsolve_system(eq, funcs=[f(x), g(x)], ics={f(0): 1, g(0): 1}) == ([], []))