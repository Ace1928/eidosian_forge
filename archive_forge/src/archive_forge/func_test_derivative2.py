from sympy import abc
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.hyper import meijerg
from sympy.polys.polytools import Poly
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import signsimp
from sympy.testing.pytest import XFAIL
def test_derivative2():
    f = Function('f')
    x = Symbol('x')
    a = Wild('a', exclude=[f, x])
    b = Wild('b', exclude=[f])
    e = Derivative(f(x), x)
    assert e.match(Derivative(f(x), x)) == {}
    assert e.match(Derivative(f(x), x, x)) is None
    e = Derivative(f(x), x, x)
    assert e.match(Derivative(f(x), x)) is None
    assert e.match(Derivative(f(x), x, x)) == {}
    e = Derivative(f(x), x) + x ** 2
    assert e.match(a * Derivative(f(x), x) + b) == {a: 1, b: x ** 2}
    assert e.match(a * Derivative(f(x), x, x) + b) is None
    e = Derivative(f(x), x, x) + x ** 2
    assert e.match(a * Derivative(f(x), x) + b) is None
    assert e.match(a * Derivative(f(x), x, x) + b) == {a: 1, b: x ** 2}