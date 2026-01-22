from sympy.core.function import (Derivative as D, Function)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.testing.pytest import raises
from sympy.calculus.euler import euler_equations as euler
def test_euler_pendulum():
    x = Function('x')
    t = Symbol('t')
    L = D(x(t), t) ** 2 / 2 + cos(x(t))
    assert euler(L, x(t), t) == [Eq(-sin(x(t)) - D(x(t), t, t), 0)]