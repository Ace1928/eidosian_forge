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
def test_TR10i():
    assert TR10i(cos(1) * cos(3) + sin(1) * sin(3)) == cos(2)
    assert TR10i(cos(1) * cos(3) - sin(1) * sin(3)) == cos(4)
    assert TR10i(cos(1) * sin(3) - sin(1) * cos(3)) == sin(2)
    assert TR10i(cos(1) * sin(3) + sin(1) * cos(3)) == sin(4)
    assert TR10i(cos(1) * sin(3) + sin(1) * cos(3) + 7) == sin(4) + 7
    assert TR10i(cos(1) * sin(3) + sin(1) * cos(3) + cos(3)) == cos(3) + sin(4)
    assert TR10i(2 * cos(1) * sin(3) + 2 * sin(1) * cos(3) + cos(3)) == 2 * sin(4) + cos(3)
    assert TR10i(cos(2) * cos(3) + sin(2) * (cos(1) * sin(2) + cos(2) * sin(1))) == cos(1)
    eq = (cos(2) * cos(3) + sin(2) * (cos(1) * sin(2) + cos(2) * sin(1))) * cos(5) + sin(1) * sin(5)
    assert TR10i(eq) == TR10i(eq.expand()) == cos(4)
    assert TR10i(sqrt(2) * cos(x) * x + sqrt(6) * sin(x) * x) == 2 * sqrt(2) * x * sin(x + pi / 6)
    assert TR10i(cos(x) / sqrt(6) + sin(x) / sqrt(2) + cos(x) / sqrt(6) / 3 + sin(x) / sqrt(2) / 3) == 4 * sqrt(6) * sin(x + pi / 6) / 9
    assert TR10i(cos(x) / sqrt(6) + sin(x) / sqrt(2) + cos(y) / sqrt(6) / 3 + sin(y) / sqrt(2) / 3) == sqrt(6) * sin(x + pi / 6) / 3 + sqrt(6) * sin(y + pi / 6) / 9
    assert TR10i(cos(x) + sqrt(3) * sin(x) + 2 * sqrt(3) * cos(x + pi / 6)) == 4 * cos(x)
    assert TR10i(cos(x) + sqrt(3) * sin(x) + 2 * sqrt(3) * cos(x + pi / 6) + 4 * sin(x)) == 4 * sqrt(2) * sin(x + pi / 4)
    assert TR10i(cos(2) * sin(3) + sin(2) * cos(4)) == sin(2) * cos(4) + sin(3) * cos(2)
    A = Symbol('A', commutative=False)
    assert TR10i(sqrt(2) * cos(x) * A + sqrt(6) * sin(x) * A) == 2 * sqrt(2) * sin(x + pi / 6) * A
    c = cos(x)
    s = sin(x)
    h = sin(y)
    r = cos(y)
    for si in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        for argsi in ((c * r, s * h), (c * h, s * r)):
            args = zip(si, argsi)
            ex = Add(*[Mul(*ai) for ai in args])
            t = TR10i(ex)
            assert not (ex - t.expand(trig=True) or t.is_Add)
    c = cos(x)
    s = sin(x)
    h = sin(pi / 6)
    r = cos(pi / 6)
    for si in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        for argsi in ((c * r, s * h), (c * h, s * r)):
            args = zip(si, argsi)
            ex = Add(*[Mul(*ai) for ai in args])
            t = TR10i(ex)
            assert not (ex - t.expand(trig=True) or t.is_Add)