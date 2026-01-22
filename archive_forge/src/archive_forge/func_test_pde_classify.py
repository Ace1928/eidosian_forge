from sympy.core.function import (Derivative as D, Function)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.core import S
from sympy.solvers.pde import (pde_separate, pde_separate_add, pde_separate_mul,
from sympy.testing.pytest import raises
def test_pde_classify():
    f = Function('f')
    eq1 = a * f(x, y) + b * f(x, y).diff(x) + c * f(x, y).diff(y)
    eq2 = 3 * f(x, y) + 2 * f(x, y).diff(x) + f(x, y).diff(y)
    eq3 = a * f(x, y) + b * f(x, y).diff(x) + 2 * f(x, y).diff(y)
    eq4 = x * f(x, y) + f(x, y).diff(x) + 3 * f(x, y).diff(y)
    eq5 = x ** 2 * f(x, y) + x * f(x, y).diff(x) + x * y * f(x, y).diff(y)
    eq6 = y * x ** 2 * f(x, y) + y * f(x, y).diff(x) + f(x, y).diff(y)
    for eq in [eq1, eq2, eq3]:
        assert classify_pde(eq) == ('1st_linear_constant_coeff_homogeneous',)
    for eq in [eq4, eq5, eq6]:
        assert classify_pde(eq) == ('1st_linear_variable_coeff',)