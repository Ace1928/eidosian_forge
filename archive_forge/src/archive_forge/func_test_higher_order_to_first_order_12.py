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
def test_higher_order_to_first_order_12():
    f, g = symbols('f g', cls=Function)
    x = symbols('x')
    x, y = symbols('x, y', cls=Function)
    t, l = symbols('t, l')
    eqs12 = [Eq(4 * x(t) + Derivative(x(t), (t, 2)) + 8 * Derivative(y(t), t), 0), Eq(4 * y(t) - 8 * Derivative(x(t), t) + Derivative(y(t), (t, 2)), 0)]
    sol12 = [Eq(y(t), C1 * (2 - sqrt(5)) * sin(2 * t * sqrt(4 * sqrt(5) + 9)) * Rational(-1, 2) + C2 * (2 - sqrt(5)) * cos(2 * t * sqrt(4 * sqrt(5) + 9)) / 2 + C3 * (2 + sqrt(5)) * sin(2 * t * sqrt(9 - 4 * sqrt(5))) * Rational(-1, 2) + C4 * (2 + sqrt(5)) * cos(2 * t * sqrt(9 - 4 * sqrt(5))) / 2), Eq(x(t), C1 * (2 - sqrt(5)) * cos(2 * t * sqrt(4 * sqrt(5) + 9)) * Rational(-1, 2) + C2 * (2 - sqrt(5)) * sin(2 * t * sqrt(4 * sqrt(5) + 9)) * Rational(-1, 2) + C3 * (2 + sqrt(5)) * cos(2 * t * sqrt(9 - 4 * sqrt(5))) / 2 + C4 * (2 + sqrt(5)) * sin(2 * t * sqrt(9 - 4 * sqrt(5))) / 2)]
    assert dsolve(eqs12) == sol12
    assert checksysodesol(eqs12, sol12) == (True, [0, 0])