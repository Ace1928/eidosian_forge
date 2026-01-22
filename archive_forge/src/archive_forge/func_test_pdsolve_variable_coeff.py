from sympy.core.function import (Derivative as D, Function)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.core import S
from sympy.solvers.pde import (pde_separate, pde_separate_add, pde_separate_mul,
from sympy.testing.pytest import raises
def test_pdsolve_variable_coeff():
    f, F = map(Function, ['f', 'F'])
    u = f(x, y)
    eq = x * u.diff(x) - y * u.diff(y) + y ** 2 * u - y ** 2
    sol = pdsolve(eq, hint='1st_linear_variable_coeff')
    assert sol == Eq(u, F(x * y) * exp(y ** 2 / 2) + 1)
    assert checkpdesol(eq, sol)[0]
    eq = x ** 2 * u + x * u.diff(x) + x * y * u.diff(y)
    sol = pdsolve(eq, hint='1st_linear_variable_coeff')
    assert sol == Eq(u, F(y * exp(-x)) * exp(-x ** 2 / 2))
    assert checkpdesol(eq, sol)[0]
    eq = y * x ** 2 * u + y * u.diff(x) + u.diff(y)
    sol = pdsolve(eq, hint='1st_linear_variable_coeff')
    assert sol == Eq(u, F(-2 * x + y ** 2) * exp(-x ** 3 / 3))
    assert checkpdesol(eq, sol)[0]
    eq = exp(x) ** 2 * u.diff(x) + y
    sol = pdsolve(eq, hint='1st_linear_variable_coeff')
    assert sol == Eq(u, y * exp(-2 * x) / 2 + F(y))
    assert checkpdesol(eq, sol)[0]
    eq = exp(2 * x) * u.diff(y) + y * u - u
    sol = pdsolve(eq, hint='1st_linear_variable_coeff')
    assert sol == Eq(u, F(x) * exp(-y * (y - 2) * exp(-2 * x) / 2))