from sympy.polys.domains.groundtypes import (
from sympy.polys.domains.integerring import IntegerRing
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public
def from_QQ(K1, a, K0):
    """Convert Python's ``Fraction`` to Python's ``int``. """
    if a.denominator == 1:
        return a.numerator