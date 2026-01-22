from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import cos
from sympy.integrals.integrals import Integral
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.sqrtdenest import (
def test_issue_12420():
    assert sqrtdenest((3 - sqrt(2) * sqrt(4 + 3 * I) + 3 * I) / 2) == I
    e = 3 - sqrt(2) * sqrt(4 + I) + 3 * I
    assert sqrtdenest(e) == e