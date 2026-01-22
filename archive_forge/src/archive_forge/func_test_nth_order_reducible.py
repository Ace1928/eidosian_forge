from sympy.core.function import (Derivative, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, cosh, sinh, tanh)
from sympy.functions.elementary.miscellaneous import (cbrt, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sec, sin, tan)
from sympy.functions.special.error_functions import (Ei, erfi)
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import (Integral, integrate)
from sympy.polys.rootoftools import rootof
from sympy.core import Function, Symbol
from sympy.functions import airyai, airybi, besselj, bessely, lowergamma
from sympy.integrals.risch import NonElementaryIntegral
from sympy.solvers.ode import classify_ode, dsolve
from sympy.solvers.ode.ode import allhints, _remove_redundant_solutions
from sympy.solvers.ode.single import (FirstLinear, ODEMatchError,
from sympy.solvers.ode.subscheck import checkodesol
from sympy.testing.pytest import raises, slow, ON_CI
import traceback
from sympy.solvers.ode.tests.test_single import _test_an_example
def test_nth_order_reducible():
    F = lambda eq: NthOrderReducible(SingleODEProblem(eq, f(x), x))._matches()
    D = Derivative
    assert F(D(y * f(x), x, y) + D(f(x), x)) == False
    assert F(D(y * f(y), y, y) + D(f(y), y)) == False
    assert F(f(x) * D(f(x), x) + D(f(x), x, 2)) == False
    assert F(D(x * f(y), y, 2) + D(u * y * f(x), x, 3)) == False
    assert F(D(f(y), y, 2) + D(f(y), y, 3) + D(f(x), x, 4)) == False
    assert F(D(f(x), x, 2) + D(f(x), x, 3)) == True
    _ode_solver_test(_get_examples_ode_sol_nth_order_reducible)