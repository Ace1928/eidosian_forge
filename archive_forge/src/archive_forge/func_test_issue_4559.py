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
def test_issue_4559():
    x = Symbol('x')
    e = Symbol('e')
    w = Wild('w', exclude=[x])
    y = Wild('y')
    assert (3 / x).match(w / y) == {w: 3, y: x}
    assert (3 * x).match(w * y) == {w: 3, y: x}
    assert (x / 3).match(y / w) == {w: 3, y: x}
    assert (3 * x).match(y / w) == {w: S.One / 3, y: x}
    assert (3 * x).match(y / w) == {w: Rational(1, 3), y: x}
    assert (x / 3).match(w / y) == {w: S.One / 3, y: 1 / x}
    assert (3 * x).match(w / y) == {w: 3, y: 1 / x}
    assert (3 / x).match(w * y) == {w: 3, y: 1 / x}
    r = Symbol('r', rational=True)
    assert (x ** r).match(y ** 2) == {y: x ** (r / 2)}
    assert (x ** e).match(y ** 2) == {y: sqrt(x ** e)}
    a = Wild('a')
    e = S.Zero
    assert e.match(a) == {a: e}
    assert e.match(1 / a) is None
    assert e.match(a ** 0.3) is None
    e = S(3)
    assert e.match(1 / a) == {a: 1 / e}
    assert e.match(1 / a ** 2) == {a: 1 / sqrt(e)}
    e = pi
    assert e.match(1 / a) == {a: 1 / e}
    assert e.match(1 / a ** 2) == {a: 1 / sqrt(e)}
    assert (-e).match(sqrt(a)) is None
    assert (-e).match(a ** 2) == {a: I * sqrt(pi)}