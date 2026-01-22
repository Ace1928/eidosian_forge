from sympy.core.numbers import (AlgebraicNumber, I, pi, Rational)
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.external.gmpy import MPQ
from sympy.polys.numberfields.subfield import (
from sympy.polys.polyerrors import IsomorphismFailed
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.testing.pytest import raises
from sympy.abc import x
def test_primitive_element():
    assert primitive_element([sqrt(2)], x) == (x ** 2 - 2, [1])
    assert primitive_element([sqrt(2), sqrt(3)], x) == (x ** 4 - 10 * x ** 2 + 1, [1, 1])
    assert primitive_element([sqrt(2)], x, polys=True) == (Poly(x ** 2 - 2, domain='QQ'), [1])
    assert primitive_element([sqrt(2), sqrt(3)], x, polys=True) == (Poly(x ** 4 - 10 * x ** 2 + 1, domain='QQ'), [1, 1])
    assert primitive_element([sqrt(2)], x, ex=True) == (x ** 2 - 2, [1], [[1, 0]])
    assert primitive_element([sqrt(2), sqrt(3)], x, ex=True) == (x ** 4 - 10 * x ** 2 + 1, [1, 1], [[Q(1, 2), 0, -Q(9, 2), 0], [-Q(1, 2), 0, Q(11, 2), 0]])
    assert primitive_element([sqrt(2)], x, ex=True, polys=True) == (Poly(x ** 2 - 2, domain='QQ'), [1], [[1, 0]])
    assert primitive_element([sqrt(2), sqrt(3)], x, ex=True, polys=True) == (Poly(x ** 4 - 10 * x ** 2 + 1, domain='QQ'), [1, 1], [[Q(1, 2), 0, -Q(9, 2), 0], [-Q(1, 2), 0, Q(11, 2), 0]])
    assert primitive_element([sqrt(2)], polys=True) == (Poly(x ** 2 - 2), [1])
    raises(ValueError, lambda: primitive_element([], x, ex=False))
    raises(ValueError, lambda: primitive_element([], x, ex=True))
    a, b = (I * sqrt(2 * sqrt(2) + 3), I * sqrt(-2 * sqrt(2) + 3))
    assert primitive_element([a, b, I], x) == (x ** 4 + 6 * x ** 2 + 1, [1, 0, 0])
    assert primitive_element([sqrt(2), 0], x) == (x ** 2 - 2, [1, 0])
    assert primitive_element([0, sqrt(2)], x) == (x ** 2 - 2, [1, 1])
    assert primitive_element([sqrt(2), 0], x, ex=True) == (x ** 2 - 2, [1, 0], [[MPQ(1, 1), MPQ(0, 1)], []])
    assert primitive_element([0, sqrt(2)], x, ex=True) == (x ** 2 - 2, [1, 1], [[], [MPQ(1, 1), MPQ(0, 1)]])