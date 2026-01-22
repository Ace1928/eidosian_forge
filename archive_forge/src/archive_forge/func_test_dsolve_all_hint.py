from sympy.core.function import (Derivative, Function, Subs, diff)
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import acosh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan2, cos, sin, tan)
from sympy.integrals.integrals import Integral
from sympy.polys.polytools import Poly
from sympy.series.order import O
from sympy.simplify.radsimp import collect
from sympy.solvers.ode import (classify_ode,
from sympy.solvers.ode.subscheck import checkodesol
from sympy.solvers.ode.ode import (classify_sysode,
from sympy.solvers.ode.nonhomogeneous import _undetermined_coefficients_match
from sympy.solvers.ode.single import LinearCoefficients
from sympy.solvers.deutils import ode_order
from sympy.testing.pytest import XFAIL, raises, slow
from sympy.utilities.misc import filldedent
def test_dsolve_all_hint():
    eq = f(x).diff(x)
    output = dsolve(eq, hint='all')
    sol1 = output['separable_Integral']
    _y = sol1.lhs.args[1][0]
    sol1 = output['1st_homogeneous_coeff_subs_dep_div_indep_Integral']
    _u1 = sol1.rhs.args[1].args[1][0]
    expected = {'Bernoulli_Integral': Eq(f(x), C1 + Integral(0, x)), '1st_homogeneous_coeff_best': Eq(f(x), C1), 'Bernoulli': Eq(f(x), C1), 'nth_algebraic': Eq(f(x), C1), 'nth_linear_euler_eq_homogeneous': Eq(f(x), C1), 'nth_linear_constant_coeff_homogeneous': Eq(f(x), C1), 'separable': Eq(f(x), C1), '1st_homogeneous_coeff_subs_indep_div_dep': Eq(f(x), C1), 'nth_algebraic_Integral': Eq(f(x), C1), '1st_linear': Eq(f(x), C1), '1st_linear_Integral': Eq(f(x), C1 + Integral(0, x)), '1st_exact': Eq(f(x), C1), '1st_exact_Integral': Eq(Subs(Integral(0, x) + Integral(1, _y), _y, f(x)), C1), 'lie_group': Eq(f(x), C1), '1st_homogeneous_coeff_subs_dep_div_indep': Eq(f(x), C1), '1st_homogeneous_coeff_subs_dep_div_indep_Integral': Eq(log(x), C1 + Integral(-1 / _u1, (_u1, f(x) / x))), '1st_power_series': Eq(f(x), C1), 'separable_Integral': Eq(Integral(1, (_y, f(x))), C1 + Integral(0, x)), '1st_homogeneous_coeff_subs_indep_div_dep_Integral': Eq(f(x), C1), 'best': Eq(f(x), C1), 'best_hint': 'nth_algebraic', 'default': 'nth_algebraic', 'order': 1}
    assert output == expected
    assert dsolve(eq, hint='best') == Eq(f(x), C1)