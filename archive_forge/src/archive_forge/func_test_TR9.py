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
def test_TR9():
    a = S.Half
    b = 3 * a
    assert TR9(a) == a
    assert TR9(cos(1) + cos(2)) == 2 * cos(a) * cos(b)
    assert TR9(cos(1) - cos(2)) == 2 * sin(a) * sin(b)
    assert TR9(sin(1) - sin(2)) == -2 * sin(a) * cos(b)
    assert TR9(sin(1) + sin(2)) == 2 * sin(b) * cos(a)
    assert TR9(cos(1) + 2 * sin(1) + 2 * sin(2)) == cos(1) + 4 * sin(b) * cos(a)
    assert TR9(cos(4) + cos(2) + 2 * cos(1) * cos(3)) == 4 * cos(1) * cos(3)
    assert TR9((cos(4) + cos(2)) / cos(3) / 2 + cos(3)) == 2 * cos(1) * cos(2)
    assert TR9(cos(3) + cos(4) + cos(5) + cos(6)) == 4 * cos(S.Half) * cos(1) * cos(Rational(9, 2))
    assert TR9(cos(3) + cos(3) * cos(2)) == cos(3) + cos(2) * cos(3)
    assert TR9(-cos(y) + cos(x * y)) == -2 * sin(x * y / 2 - y / 2) * sin(x * y / 2 + y / 2)
    assert TR9(-sin(y) + sin(x * y)) == 2 * sin(x * y / 2 - y / 2) * cos(x * y / 2 + y / 2)
    c = cos(x)
    s = sin(x)
    for si in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        for a in ((c, s), (s, c), (cos(x), cos(x * y)), (sin(x), sin(x * y))):
            args = zip(si, a)
            ex = Add(*[Mul(*ai) for ai in args])
            t = TR9(ex)
            assert not (a[0].func == a[1].func and (not verify_numerically(ex, t.expand(trig=True)) or t.is_Add) or (a[1].func != a[0].func and ex != t))