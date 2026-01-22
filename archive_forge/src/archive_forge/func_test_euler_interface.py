from sympy.core.function import (Derivative as D, Function)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.testing.pytest import raises
from sympy.calculus.euler import euler_equations as euler
def test_euler_interface():
    x = Function('x')
    y = Symbol('y')
    t = Symbol('t')
    raises(TypeError, lambda: euler())
    raises(TypeError, lambda: euler(D(x(t), t) * y(t), [x(t), y]))
    raises(ValueError, lambda: euler(D(x(t), t) * x(y), [x(t), x(y)]))
    raises(TypeError, lambda: euler(D(x(t), t) ** 2, x(0)))
    raises(TypeError, lambda: euler(D(x(t), t) * y(t), [t]))
    assert euler(D(x(t), t) ** 2 / 2, {x(t)}) == [Eq(-D(x(t), t, t), 0)]
    assert euler(D(x(t), t) ** 2 / 2, x(t), {t}) == [Eq(-D(x(t), t, t), 0)]