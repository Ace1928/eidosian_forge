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
def test_TR8():
    assert TR8(cos(2) * cos(3)) == cos(5) / 2 + cos(1) / 2
    assert TR8(cos(2) * sin(3)) == sin(5) / 2 + sin(1) / 2
    assert TR8(sin(2) * sin(3)) == -cos(5) / 2 + cos(1) / 2
    assert TR8(sin(1) * sin(2) * sin(3)) == sin(4) / 4 - sin(6) / 4 + sin(2) / 4
    assert TR8(cos(2) * cos(3) * cos(4) * cos(5)) == cos(4) / 4 + cos(10) / 8 + cos(2) / 8 + cos(8) / 8 + cos(14) / 8 + cos(6) / 8 + Rational(1, 8)
    assert TR8(cos(2) * cos(3) * cos(4) * cos(5) * cos(6)) == cos(10) / 8 + cos(4) / 8 + 3 * cos(2) / 16 + cos(16) / 16 + cos(8) / 8 + cos(14) / 16 + cos(20) / 16 + cos(12) / 16 + Rational(1, 16) + cos(6) / 8
    assert TR8(sin(pi * Rational(3, 7)) ** 2 * cos(pi * Rational(3, 7)) ** 2 / (16 * sin(pi / 7) ** 2)) == Rational(1, 64)