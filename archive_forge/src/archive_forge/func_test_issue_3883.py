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
def test_issue_3883():
    from sympy.abc import gamma, mu, x
    f = (-gamma * (x - mu) ** 2 - log(gamma) + log(2 * pi)) / 2
    a, b, c = symbols('a b c', cls=Wild, exclude=(gamma,))
    assert f.match(a * log(gamma) + b * gamma + c) == {a: Rational(-1, 2), b: -(-mu + x) ** 2 / 2, c: log(2 * pi) / 2}
    assert f.expand().collect(gamma).match(a * log(gamma) + b * gamma + c) == {a: Rational(-1, 2), b: (-(x - mu) ** 2 / 2).expand(), c: (log(2 * pi) / 2).expand()}
    g1 = Wild('g1', exclude=[gamma])
    g2 = Wild('g2', exclude=[gamma])
    g3 = Wild('g3', exclude=[gamma])
    assert f.expand().match(g1 * log(gamma) + g2 * gamma + g3) == {g3: log(2) / 2 + log(pi) / 2, g1: Rational(-1, 2), g2: -mu ** 2 / 2 + mu * x - x ** 2 / 2}