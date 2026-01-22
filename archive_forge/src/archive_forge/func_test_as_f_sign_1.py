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
def test_as_f_sign_1():
    assert as_f_sign_1(x + 1) == (1, x, 1)
    assert as_f_sign_1(x - 1) == (1, x, -1)
    assert as_f_sign_1(-x + 1) == (-1, x, -1)
    assert as_f_sign_1(-x - 1) == (-1, x, 1)
    assert as_f_sign_1(2 * x + 2) == (2, x, 1)
    assert as_f_sign_1(x * y - y) == (y, x, -1)
    assert as_f_sign_1(-x * y + y) == (-y, x, -1)