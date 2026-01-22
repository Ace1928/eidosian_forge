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
def test_TRpower():
    assert TRpower(1 / sin(x) ** 2) == 1 / sin(x) ** 2
    assert TRpower(cos(x) ** 3 * sin(x / 2) ** 4) == (3 * cos(x) / 4 + cos(3 * x) / 4) * (-cos(x) / 2 + cos(2 * x) / 8 + Rational(3, 8))
    for k in range(2, 8):
        assert verify_numerically(sin(x) ** k, TRpower(sin(x) ** k))
        assert verify_numerically(cos(x) ** k, TRpower(cos(x) ** k))