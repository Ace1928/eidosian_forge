from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import cos
from sympy.integrals.integrals import Integral
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.sqrtdenest import (
def test_sqrt_ratcomb():
    assert sqrtdenest(sqrt(1 + r3) + sqrt(3 + 3 * r3) - sqrt(10 + 6 * r3)) == 0