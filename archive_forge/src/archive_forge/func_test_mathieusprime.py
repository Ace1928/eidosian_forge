from sympy.core.function import diff
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.mathieu_functions import (mathieuc, mathieucprime, mathieus, mathieusprime)
from sympy.abc import a, q, z
def test_mathieusprime():
    assert isinstance(mathieusprime(a, q, z), mathieusprime)
    assert mathieusprime(a, 0, z) == sqrt(a) * cos(sqrt(a) * z)
    assert diff(mathieusprime(a, q, z), z) == (-a + 2 * q * cos(2 * z)) * mathieus(a, q, z)