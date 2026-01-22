from sympy.core.add import Add
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys.polytools import factor
from sympy.series.order import O
from sympy.simplify.radsimp import (collect, collect_const, fraction, radsimp, rcollect)
from sympy.core.expr import unchanged
from sympy.core.mul import _unevaluated_Mul as umul
from sympy.simplify.radsimp import (_unevaluated_Add,
from sympy.testing.pytest import raises
from sympy.abc import x, y, z, a, b, c, d
def test_collect_Wild():
    """Collect with respect to functions with Wild argument"""
    a, b, x, y = symbols('a b x y')
    f = Function('f')
    w1 = Wild('.1')
    w2 = Wild('.2')
    assert collect(f(x) + a * f(x), f(w1)) == (1 + a) * f(x)
    assert collect(f(x, y) + a * f(x, y), f(w1)) == f(x, y) + a * f(x, y)
    assert collect(f(x, y) + a * f(x, y), f(w1, w2)) == (1 + a) * f(x, y)
    assert collect(f(x, y) + a * f(x, y), f(w1, w1)) == f(x, y) + a * f(x, y)
    assert collect(f(x, x) + a * f(x, x), f(w1, w1)) == (1 + a) * f(x, x)
    assert collect(a * (x + 1) ** y + (x + 1) ** y, w1 ** y) == (1 + a) * (x + 1) ** y
    assert collect(a * (x + 1) ** y + (x + 1) ** y, w1 ** b) == a * (x + 1) ** y + (x + 1) ** y
    assert collect(a * (x + 1) ** y + (x + 1) ** y, (x + 1) ** w2) == (1 + a) * (x + 1) ** y
    assert collect(a * (x + 1) ** y + (x + 1) ** y, w1 ** w2) == (1 + a) * (x + 1) ** y