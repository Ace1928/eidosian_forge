from sympy.polys.polyfuncs import (
from sympy.polys.polyerrors import (
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.testing.pytest import raises
from sympy.abc import a, b, c, d, e, x, y, z
def test_horner():
    assert horner(0) == 0
    assert horner(1) == 1
    assert horner(x) == x
    assert horner(x + 1) == x + 1
    assert horner(x ** 2 + 1) == x ** 2 + 1
    assert horner(x ** 2 + x) == (x + 1) * x
    assert horner(x ** 2 + x + 1) == (x + 1) * x + 1
    assert horner(9 * x ** 4 + 8 * x ** 3 + 7 * x ** 2 + 6 * x + 5) == (((9 * x + 8) * x + 7) * x + 6) * x + 5
    assert horner(a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e) == (((a * x + b) * x + c) * x + d) * x + e
    assert horner(4 * x ** 2 * y ** 2 + 2 * x ** 2 * y + 2 * x * y ** 2 + x * y, wrt=x) == ((4 * y + 2) * x * y + (2 * y + 1) * y) * x
    assert horner(4 * x ** 2 * y ** 2 + 2 * x ** 2 * y + 2 * x * y ** 2 + x * y, wrt=y) == ((4 * x + 2) * y * x + (2 * x + 1) * x) * y