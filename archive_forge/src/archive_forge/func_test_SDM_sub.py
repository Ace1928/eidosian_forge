from itertools import product
from sympy.core.singleton import S
from sympy.external.gmpy import HAS_GMPY
from sympy.testing.pytest import raises
from sympy.polys.domains import QQ, ZZ, EXRAW
from sympy.polys.matrices.sdm import SDM
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (DMBadInputError, DMDomainError,
def test_SDM_sub():
    A = SDM({0: {1: ZZ(1)}, 1: {0: ZZ(2), 1: ZZ(3)}}, (2, 2), ZZ)
    B = SDM({0: {0: ZZ(1)}, 1: {0: ZZ(-2), 1: ZZ(3)}}, (2, 2), ZZ)
    C = SDM({0: {0: ZZ(-1), 1: ZZ(1)}, 1: {0: ZZ(4)}}, (2, 2), ZZ)
    assert A.sub(B) == A - B == C
    raises(TypeError, lambda: A - [])