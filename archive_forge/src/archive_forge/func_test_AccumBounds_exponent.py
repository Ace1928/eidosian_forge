from sympy.core.numbers import (E, Rational, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core import Add, Mul, Pow
from sympy.core.expr import unchanged
from sympy.testing.pytest import raises, XFAIL
from sympy.abc import x
def test_AccumBounds_exponent():
    z = 0 ** B(a, a + S.Half)
    assert z.subs(a, 0) == B(0, 1)
    assert z.subs(a, 1) == 0
    p = z.subs(a, -1)
    assert p.is_Pow and p.args == (0, B(-1, -S.Half))
    assert 1 ** B(a, a + 1) == 1
    assert S.Half ** B(-2, 2) == B(S(1) / 4, 4)
    assert 2 ** B(-2, 2) == B(S(1) / 4, 4)
    assert B(0, 1) ** B(S(1) / 2, 1) == B(0, 1)
    assert B(0, 1) ** B(0, 1) == B(0, 1)
    assert B(2, 3) ** B(-3, -2) == B(S(1) / 27, S(1) / 4)
    assert B(2, 3) ** B(-3, 2) == B(S(1) / 27, 9)
    assert unchanged(Pow, B(-1, 1), B(1, 2))
    assert B(0, S(1) / 2) ** B(1, oo) == B(0, S(1) / 2)
    assert B(0, 1) ** B(1, oo) == B(0, oo)
    assert B(0, 2) ** B(1, oo) == B(0, oo)
    assert B(0, oo) ** B(1, oo) == B(0, oo)
    assert B(S(1) / 2, 1) ** B(1, oo) == B(0, oo)
    assert B(S(1) / 2, 1) ** B(-oo, -1) == B(0, oo)
    assert B(S(1) / 2, 1) ** B(-oo, oo) == B(0, oo)
    assert B(S(1) / 2, 2) ** B(1, oo) == B(0, oo)
    assert B(S(1) / 2, 2) ** B(-oo, -1) == B(0, oo)
    assert B(S(1) / 2, 2) ** B(-oo, oo) == B(0, oo)
    assert B(S(1) / 2, oo) ** B(1, oo) == B(0, oo)
    assert B(S(1) / 2, oo) ** B(-oo, -1) == B(0, oo)
    assert B(S(1) / 2, oo) ** B(-oo, oo) == B(0, oo)
    assert B(1, 2) ** B(1, oo) == B(0, oo)
    assert B(1, 2) ** B(-oo, -1) == B(0, oo)
    assert B(1, 2) ** B(-oo, oo) == B(0, oo)
    assert B(1, oo) ** B(1, oo) == B(0, oo)
    assert B(1, oo) ** B(-oo, -1) == B(0, oo)
    assert B(1, oo) ** B(-oo, oo) == B(0, oo)
    assert B(2, oo) ** B(1, oo) == B(2, oo)
    assert B(2, oo) ** B(-oo, -1) == B(0, S(1) / 2)
    assert B(2, oo) ** B(-oo, oo) == B(0, oo)