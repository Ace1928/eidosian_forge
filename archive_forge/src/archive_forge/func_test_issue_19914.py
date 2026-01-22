from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import cos
from sympy.integrals.integrals import Integral
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.sqrtdenest import (
def test_issue_19914():
    a = Integer(-8)
    b = Integer(-1)
    r = Integer(63)
    d2 = a * a - b * b * r
    assert _sqrt_numeric_denest(a, b, r, d2) == sqrt(14) * I / 2 + 3 * sqrt(2) * I / 2
    assert sqrtdenest(sqrt(-8 - sqrt(63))) == sqrt(14) * I / 2 + 3 * sqrt(2) * I / 2