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
def test_higher_order1_slow1():
    x, y = symbols('x y', cls=Function)
    t = symbols('t')
    eq = [Eq(diff(x(t), t, t), (log(t) + t ** 2) * diff(x(t), t) + (log(t) + t ** 2) * 3 * diff(y(t), t)), Eq(diff(y(t), t, t), (log(t) + t ** 2) * 2 * diff(x(t), t) + (log(t) + t ** 2) * 9 * diff(y(t), t))]
    sol, = dsolve_system(eq, simplify=False, doit=False)
    for e in eq:
        res = (e.lhs - e.rhs).subs({sol[0].lhs: sol[0].rhs, sol[1].lhs: sol[1].rhs})
        res = res.subs({d: d.doit(deep=False) for d in res.atoms(Derivative)})
        assert ratsimp(res.subs(t, 1)) == 0