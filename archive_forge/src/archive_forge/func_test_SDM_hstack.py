from itertools import product
from sympy.core.singleton import S
from sympy.external.gmpy import HAS_GMPY
from sympy.testing.pytest import raises
from sympy.polys.domains import QQ, ZZ, EXRAW
from sympy.polys.matrices.sdm import SDM
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (DMBadInputError, DMDomainError,
def test_SDM_hstack():
    A = SDM({0: {1: ZZ(1)}}, (2, 2), ZZ)
    B = SDM({1: {1: ZZ(1)}}, (2, 2), ZZ)
    AA = SDM({0: {1: ZZ(1), 3: ZZ(1)}}, (2, 4), ZZ)
    AB = SDM({0: {1: ZZ(1)}, 1: {3: ZZ(1)}}, (2, 4), ZZ)
    assert SDM.hstack(A) == A
    assert SDM.hstack(A, A) == AA
    assert SDM.hstack(A, B) == AB