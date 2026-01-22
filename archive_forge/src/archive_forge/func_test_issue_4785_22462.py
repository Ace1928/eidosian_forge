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
def test_issue_4785_22462():
    from sympy.abc import A
    eq = x + A * (x + diff(f(x), x) + f(x)) + diff(f(x), x) + f(x) + 2
    assert classify_ode(eq, f(x)) == ('factorable', '1st_exact', '1st_linear', 'Bernoulli', 'almost_linear', '1st_power_series', 'lie_group', 'nth_linear_constant_coeff_undetermined_coefficients', 'nth_linear_constant_coeff_variation_of_parameters', '1st_exact_Integral', '1st_linear_Integral', 'Bernoulli_Integral', 'almost_linear_Integral', 'nth_linear_constant_coeff_variation_of_parameters_Integral')
    eq = (x ** 2 + f(x) ** 2) * f(x).diff(x) - 2 * x * f(x)
    assert classify_ode(eq, f(x)) == ('factorable', '1st_exact', '1st_homogeneous_coeff_best', '1st_homogeneous_coeff_subs_indep_div_dep', '1st_homogeneous_coeff_subs_dep_div_indep', '1st_power_series', 'lie_group', '1st_exact_Integral', '1st_homogeneous_coeff_subs_indep_div_dep_Integral', '1st_homogeneous_coeff_subs_dep_div_indep_Integral')