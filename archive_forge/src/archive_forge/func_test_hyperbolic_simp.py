from itertools import product
from sympy.core.function import (Subs, count_ops, diff, expand)
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (cosh, coth, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, cot, sin, tan)
from sympy.functions.elementary.trigonometric import (acos, asin, atan2)
from sympy.functions.elementary.trigonometric import (asec, acsc)
from sympy.functions.elementary.trigonometric import (acot, atan)
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import (exptrigsimp, trigsimp)
from sympy.testing.pytest import XFAIL
from sympy.abc import x, y
def test_hyperbolic_simp():
    x, y = symbols('x,y')
    assert trigsimp(sinh(x) ** 2 + 1) == cosh(x) ** 2
    assert trigsimp(cosh(x) ** 2 - 1) == sinh(x) ** 2
    assert trigsimp(cosh(x) ** 2 - sinh(x) ** 2) == 1
    assert trigsimp(1 - tanh(x) ** 2) == 1 / cosh(x) ** 2
    assert trigsimp(1 - 1 / cosh(x) ** 2) == tanh(x) ** 2
    assert trigsimp(tanh(x) ** 2 + 1 / cosh(x) ** 2) == 1
    assert trigsimp(coth(x) ** 2 - 1) == 1 / sinh(x) ** 2
    assert trigsimp(1 / sinh(x) ** 2 + 1) == 1 / tanh(x) ** 2
    assert trigsimp(coth(x) ** 2 - 1 / sinh(x) ** 2) == 1
    assert trigsimp(5 * cosh(x) ** 2 - 5 * sinh(x) ** 2) == 5
    assert trigsimp(5 * cosh(x / 2) ** 2 - 2 * sinh(x / 2) ** 2) == 3 * cosh(x) / 2 + Rational(7, 2)
    assert trigsimp(sinh(x) / cosh(x)) == tanh(x)
    assert trigsimp(tanh(x)) == trigsimp(sinh(x) / cosh(x))
    assert trigsimp(cosh(x) / sinh(x)) == 1 / tanh(x)
    assert trigsimp(2 * tanh(x) * cosh(x)) == 2 * sinh(x)
    assert trigsimp(coth(x) ** 3 * sinh(x) ** 3) == cosh(x) ** 3
    assert trigsimp(y * tanh(x) ** 2 / sinh(x) ** 2) == y / cosh(x) ** 2
    assert trigsimp(coth(x) / cosh(x)) == 1 / sinh(x)
    for a in (pi / 6 * I, pi / 4 * I, pi / 3 * I):
        assert trigsimp(sinh(a) * cosh(x) + cosh(a) * sinh(x)) == sinh(x + a)
        assert trigsimp(-sinh(a) * cosh(x) + cosh(a) * sinh(x)) == sinh(x - a)
    e = 2 * cosh(x) ** 2 - 2 * sinh(x) ** 2
    assert trigsimp(log(e)) == log(2)
    assert trigsimp(sqrt(cosh(x) ** 2 - 1)) == sqrt(sinh(x) ** 2)
    assert trigsimp(cosh(x) ** 2 * cosh(y) ** 2 - cosh(x) ** 2 * sinh(y) ** 2 - sinh(x) ** 2, recursive=True) == 1
    assert trigsimp(sinh(x) ** 2 * sinh(y) ** 2 - sinh(x) ** 2 * cosh(y) ** 2 + cosh(x) ** 2, recursive=True) == 1
    assert abs(trigsimp(2.0 * cosh(x) ** 2 - 2.0 * sinh(x) ** 2) - 2.0) < 1e-10
    assert trigsimp(sinh(x) ** 2 / cosh(x) ** 2) == tanh(x) ** 2
    assert trigsimp(sinh(x) ** 3 / cosh(x) ** 3) == tanh(x) ** 3
    assert trigsimp(sinh(x) ** 10 / cosh(x) ** 10) == tanh(x) ** 10
    assert trigsimp(cosh(x) ** 3 / sinh(x) ** 3) == 1 / tanh(x) ** 3
    assert trigsimp(cosh(x) / sinh(x)) == 1 / tanh(x)
    assert trigsimp(cosh(x) ** 2 / sinh(x) ** 2) == 1 / tanh(x) ** 2
    assert trigsimp(cosh(x) ** 10 / sinh(x) ** 10) == 1 / tanh(x) ** 10
    assert trigsimp(x * cosh(x) * tanh(x)) == x * sinh(x)
    assert trigsimp(-sinh(x) + cosh(x) * tanh(x)) == 0
    assert tan(x) != 1 / cot(x)
    assert trigsimp(tan(x) - 1 / cot(x)) == 0
    assert trigsimp(3 * tanh(x) ** 7 - 2 / coth(x) ** 7) == tanh(x) ** 7