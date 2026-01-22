from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import cos
from sympy.integrals.integrals import Integral
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.sqrtdenest import (
def test_sqrtdenest():
    d = {sqrt(5 + 2 * r6): r2 + r3, sqrt(5.0 + 2 * r6): sqrt(5.0 + 2 * r6), sqrt(5.0 + 4 * sqrt(5 + 2 * r6)): sqrt(5.0 + 4 * r2 + 4 * r3), sqrt(r2): sqrt(r2), sqrt(5 + r7): sqrt(5 + r7), sqrt(3 + sqrt(5 + 2 * r7)): 3 * r2 * (5 + 2 * r7) ** Rational(1, 4) / (2 * sqrt(6 + 3 * r7)) + r2 * sqrt(6 + 3 * r7) / (2 * (5 + 2 * r7) ** Rational(1, 4)), sqrt(3 + 2 * r3): 3 ** Rational(3, 4) * (r6 / 2 + 3 * r2 / 2) / 3}
    for i in d:
        assert sqrtdenest(i) == d[i], i