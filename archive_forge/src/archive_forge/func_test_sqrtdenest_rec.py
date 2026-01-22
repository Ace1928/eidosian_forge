from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import cos
from sympy.integrals.integrals import Integral
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.sqrtdenest import (
def test_sqrtdenest_rec():
    assert sqrtdenest(sqrt(-4 * sqrt(14) - 2 * r6 + 4 * sqrt(21) + 33)) == -r2 + r3 + 2 * r7
    assert sqrtdenest(sqrt(-28 * r7 - 14 * r5 + 4 * sqrt(35) + 82)) == -7 + r5 + 2 * r7
    assert sqrtdenest(sqrt(6 * r2 / 11 + 2 * sqrt(22) / 11 + 6 * sqrt(11) / 11 + 2)) == sqrt(11) * (r2 + 3 + sqrt(11)) / 11
    assert sqrtdenest(sqrt(468 * r3 + 3024 * r2 + 2912 * r6 + 19735)) == 9 * r3 + 26 + 56 * r6
    z = sqrt(-490 * r3 - 98 * sqrt(115) - 98 * sqrt(345) - 2107)
    assert sqrtdenest(z) == sqrt(-1) * (7 * r5 + 7 * r15 + 7 * sqrt(23))
    z = sqrt(-4 * sqrt(14) - 2 * r6 + 4 * sqrt(21) + 34)
    assert sqrtdenest(z) == z
    assert sqrtdenest(sqrt(-8 * r2 - 2 * r5 + 18)) == -r10 + 1 + r2 + r5
    assert sqrtdenest(sqrt(8 * r2 + 2 * r5 - 18)) == sqrt(-1) * (-r10 + 1 + r2 + r5)
    assert sqrtdenest(sqrt(8 * r2 / 3 + 14 * r5 / 3 + Rational(154, 9))) == -r10 / 3 + r2 + r5 + 3
    assert sqrtdenest(sqrt(sqrt(2 * r6 + 5) + sqrt(2 * r7 + 8))) == sqrt(1 + r2 + r3 + r7)
    assert sqrtdenest(sqrt(4 * r15 + 8 * r5 + 12 * r3 + 24)) == 1 + r3 + r5 + r15
    w = 1 + r2 + r3 + r5 + r7
    assert sqrtdenest(sqrt((w ** 2).expand())) == w
    z = sqrt((w ** 2).expand() + 1)
    assert sqrtdenest(z) == z
    z = sqrt(2 * r10 + 6 * r2 + 4 * r5 + 12 + 10 * r15 + 30 * r3)
    assert sqrtdenest(z) == z