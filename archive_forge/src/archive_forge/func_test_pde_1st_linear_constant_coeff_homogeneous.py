from sympy.core.function import (Derivative as D, Function)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.core import S
from sympy.solvers.pde import (pde_separate, pde_separate_add, pde_separate_mul,
from sympy.testing.pytest import raises
def test_pde_1st_linear_constant_coeff_homogeneous():
    f, F = map(Function, ['f', 'F'])
    u = f(x, y)
    eq = 2 * u + u.diff(x) + u.diff(y)
    assert classify_pde(eq) == ('1st_linear_constant_coeff_homogeneous',)
    sol = pdsolve(eq)
    assert sol == Eq(u, F(x - y) * exp(-x - y))
    assert checkpdesol(eq, sol)[0]
    eq = 4 + 3 * u.diff(x) / u + 2 * u.diff(y) / u
    assert classify_pde(eq) == ('1st_linear_constant_coeff_homogeneous',)
    sol = pdsolve(eq)
    assert sol == Eq(u, F(2 * x - 3 * y) * exp(-S(12) * x / 13 - S(8) * y / 13))
    assert checkpdesol(eq, sol)[0]
    eq = u + 6 * u.diff(x) + 7 * u.diff(y)
    assert classify_pde(eq) == ('1st_linear_constant_coeff_homogeneous',)
    sol = pdsolve(eq)
    assert sol == Eq(u, F(7 * x - 6 * y) * exp(-6 * x / S(85) - 7 * y / S(85)))
    assert checkpdesol(eq, sol)[0]
    eq = a * u + b * u.diff(x) + c * u.diff(y)
    sol = pdsolve(eq)
    assert checkpdesol(eq, sol)[0]