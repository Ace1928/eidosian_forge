from itertools import product
from sympy.core.singleton import S
from sympy.external.gmpy import HAS_GMPY
from sympy.testing.pytest import raises
from sympy.polys.domains import QQ, ZZ, EXRAW
from sympy.polys.matrices.sdm import SDM
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (DMBadInputError, DMDomainError,
def test_SDM_mul():
    A = SDM({0: {0: ZZ(2)}}, (2, 2), ZZ)
    B = SDM({0: {0: ZZ(4)}}, (2, 2), ZZ)
    assert A * ZZ(2) == B
    assert ZZ(2) * A == B
    raises(TypeError, lambda: A * QQ(1, 2))
    raises(TypeError, lambda: QQ(1, 2) * A)