from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import cos
from sympy.integrals.integrals import Integral
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.sqrtdenest import (
def test_sqrtdenest2():
    assert sqrtdenest(sqrt(16 - 2 * r29 + 2 * sqrt(55 - 10 * r29))) == r5 + sqrt(11 - 2 * r29)
    e = sqrt(-r5 + sqrt(-2 * r29 + 2 * sqrt(-10 * r29 + 55) + 16))
    assert sqrtdenest(e) == root(-2 * r29 + 11, 4)
    r = sqrt(1 + r7)
    assert sqrtdenest(sqrt(1 + r)) == sqrt(1 + r)
    e = sqrt(((1 + sqrt(1 + 2 * sqrt(3 + r2 + r5))) ** 2).expand())
    assert sqrtdenest(e) == 1 + sqrt(1 + 2 * sqrt(r2 + r5 + 3))
    assert sqrtdenest(sqrt(5 * r3 + 6 * r2)) == sqrt(2) * root(3, 4) + root(3, 4) ** 3
    assert sqrtdenest(sqrt(((1 + r5 + sqrt(1 + r3)) ** 2).expand())) == 1 + r5 + sqrt(1 + r3)
    assert sqrtdenest(sqrt(((1 + r5 + r7 + sqrt(1 + r3)) ** 2).expand())) == 1 + sqrt(1 + r3) + r5 + r7
    e = sqrt(((1 + cos(2) + cos(3) + sqrt(1 + r3)) ** 2).expand())
    assert sqrtdenest(e) == cos(3) + cos(2) + 1 + sqrt(1 + r3)
    e = sqrt(-2 * r10 + 2 * r2 * sqrt(-2 * r10 + 11) + 14)
    assert sqrtdenest(e) == sqrt(-2 * r10 - 2 * r2 + 4 * r5 + 14)
    z = sqrt(-2 * r29 + cos(2) + 2 * sqrt(-10 * r29 + 55) + 16)
    assert sqrtdenest(z) == z
    assert sqrtdenest(sqrt(r6 + sqrt(15))) == sqrt(r6 + sqrt(15))
    z = sqrt(15 - 2 * sqrt(31) + 2 * sqrt(55 - 10 * r29))
    assert sqrtdenest(z) == z