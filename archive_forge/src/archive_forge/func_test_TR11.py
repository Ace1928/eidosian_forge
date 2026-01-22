from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.hyperbolic import (cosh, coth, csch, sech, sinh, tanh)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (cos, cot, csc, sec, sin, tan)
from sympy.simplify.powsimp import powsimp
from sympy.simplify.fu import (
from sympy.core.random import verify_numerically
from sympy.abc import a, b, c, x, y, z
def test_TR11():
    assert TR11(sin(2 * x)) == 2 * sin(x) * cos(x)
    assert TR11(sin(4 * x)) == 4 * ((-sin(x) ** 2 + cos(x) ** 2) * sin(x) * cos(x))
    assert TR11(sin(x * Rational(4, 3))) == 4 * ((-sin(x / 3) ** 2 + cos(x / 3) ** 2) * sin(x / 3) * cos(x / 3))
    assert TR11(cos(2 * x)) == -sin(x) ** 2 + cos(x) ** 2
    assert TR11(cos(4 * x)) == (-sin(x) ** 2 + cos(x) ** 2) ** 2 - 4 * sin(x) ** 2 * cos(x) ** 2
    assert TR11(cos(2)) == cos(2)
    assert TR11(cos(pi * Rational(3, 7)), pi * Rational(2, 7)) == -cos(pi * Rational(2, 7)) ** 2 + sin(pi * Rational(2, 7)) ** 2
    assert TR11(cos(4), 2) == -sin(2) ** 2 + cos(2) ** 2
    assert TR11(cos(6), 2) == cos(6)
    assert TR11(sin(x) / cos(x / 2), x / 2) == 2 * sin(x / 2)