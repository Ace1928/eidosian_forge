from sympy.core.function import (Derivative as D, Function)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.testing.pytest import raises
from sympy.calculus.euler import euler_equations as euler
def test_euler_henonheiles():
    x = Function('x')
    y = Function('y')
    t = Symbol('t')
    L = sum((D(z(t), t) ** 2 / 2 - z(t) ** 2 / 2 for z in [x, y]))
    L += -x(t) ** 2 * y(t) + y(t) ** 3 / 3
    assert euler(L, [x(t), y(t)], t) == [Eq(-2 * x(t) * y(t) - x(t) - D(x(t), t, t), 0), Eq(-x(t) ** 2 + y(t) ** 2 - y(t) - D(y(t), t, t), 0)]