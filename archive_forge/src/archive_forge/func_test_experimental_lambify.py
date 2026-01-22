from sympy.core.symbol import symbols, Symbol
from sympy.functions import Max
from sympy.plotting.experimental_lambdify import experimental_lambdify
from sympy.plotting.intervalmath.interval_arithmetic import \
def test_experimental_lambify():
    x = Symbol('x')
    f = experimental_lambdify([x], Max(x, 5))
    assert Max(2, 5) == 5
    assert Max(5, 7) == 7
    x = Symbol('x-3')
    f = experimental_lambdify([x], x + 1)
    assert f(1) == 2