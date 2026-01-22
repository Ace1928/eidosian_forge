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
def test_fu():
    assert fu(sin(50) ** 2 + cos(50) ** 2 + sin(pi / 6)) == Rational(3, 2)
    assert fu(sqrt(6) * cos(x) + sqrt(2) * sin(x)) == 2 * sqrt(2) * sin(x + pi / 3)
    eq = sin(x) ** 4 - cos(y) ** 2 + sin(y) ** 2 + 2 * cos(x) ** 2
    assert fu(eq) == cos(x) ** 4 - 2 * cos(y) ** 2 + 2
    assert fu(S.Half - cos(2 * x) / 2) == sin(x) ** 2
    assert fu(sin(a) * (cos(b) - sin(b)) + cos(a) * (sin(b) + cos(b))) == sqrt(2) * sin(a + b + pi / 4)
    assert fu(sqrt(3) * cos(x) / 2 + sin(x) / 2) == sin(x + pi / 3)
    assert fu(1 - sin(2 * x) ** 2 / 4 - sin(y) ** 2 - cos(x) ** 4) == -cos(x) ** 2 + cos(y) ** 2
    assert fu(cos(pi * Rational(4, 9))) == sin(pi / 18)
    assert fu(cos(pi / 9) * cos(pi * Rational(2, 9)) * cos(pi * Rational(3, 9)) * cos(pi * Rational(4, 9))) == Rational(1, 16)
    assert fu(tan(pi * Rational(7, 18)) + tan(pi * Rational(5, 18)) - sqrt(3) * tan(pi * Rational(5, 18)) * tan(pi * Rational(7, 18))) == -sqrt(3)
    assert fu(tan(1) * tan(2)) == tan(1) * tan(2)
    expr = Mul(*[cos(2 ** i) for i in range(10)])
    assert fu(expr) == sin(1024) / (1024 * sin(1))
    assert fu(cos(x) + sqrt(sin(x) ** 2)) == cos(x) + sqrt(sin(x) ** 2)
    assert fu((-14 * sin(x) ** 3 + 35 * sin(x) + 6 * sqrt(3) * cos(x) ** 3 + 9 * sqrt(3) * cos(x)) / (cos(2 * x) + 4)) == 7 * sin(x) + 3 * sqrt(3) * cos(x)