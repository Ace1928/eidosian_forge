from itertools import product
from sympy.core.singleton import S
from sympy.external.gmpy import HAS_GMPY
from sympy.testing.pytest import raises
from sympy.polys.domains import QQ, ZZ, EXRAW
from sympy.polys.matrices.sdm import SDM
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (DMBadInputError, DMDomainError,
def test_SDM_setitem():
    A = SDM({0: {1: ZZ(1)}}, (2, 2), ZZ)
    A.setitem(0, 0, ZZ(1))
    assert A == SDM({0: {0: ZZ(1), 1: ZZ(1)}}, (2, 2), ZZ)
    A.setitem(1, 0, ZZ(1))
    assert A == SDM({0: {0: ZZ(1), 1: ZZ(1)}, 1: {0: ZZ(1)}}, (2, 2), ZZ)
    A.setitem(1, 0, ZZ(0))
    assert A == SDM({0: {0: ZZ(1), 1: ZZ(1)}}, (2, 2), ZZ)
    A.setitem(1, 0, ZZ(0))
    assert A == SDM({0: {0: ZZ(1), 1: ZZ(1)}}, (2, 2), ZZ)
    A.setitem(0, 0, ZZ(0))
    assert A == SDM({0: {1: ZZ(1)}}, (2, 2), ZZ)
    A.setitem(0, 0, ZZ(0))
    assert A == SDM({0: {1: ZZ(1)}}, (2, 2), ZZ)
    raises(IndexError, lambda: A.setitem(2, 0, ZZ(1)))
    raises(IndexError, lambda: A.setitem(0, 2, ZZ(1)))