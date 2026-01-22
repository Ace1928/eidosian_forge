from sympy.core.function import (Derivative as D, Function)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.core import S
from sympy.solvers.pde import (pde_separate, pde_separate_add, pde_separate_mul,
from sympy.testing.pytest import raises
def test_pde_separate():
    x, y, z, t = symbols('x,y,z,t')
    F, T, X, Y, Z, u = map(Function, 'FTXYZu')
    eq = Eq(D(u(x, t), x), D(u(x, t), t) * exp(u(x, t)))
    raises(ValueError, lambda: pde_separate(eq, u(x, t), [X(x), T(t)], 'div'))