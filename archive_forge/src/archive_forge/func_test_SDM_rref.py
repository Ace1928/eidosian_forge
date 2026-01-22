from itertools import product
from sympy.core.singleton import S
from sympy.external.gmpy import HAS_GMPY
from sympy.testing.pytest import raises
from sympy.polys.domains import QQ, ZZ, EXRAW
from sympy.polys.matrices.sdm import SDM
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (DMBadInputError, DMDomainError,
def test_SDM_rref():
    eye2 = SDM({0: {0: QQ(1)}, 1: {1: QQ(1)}}, (2, 2), QQ)
    A = SDM({0: {0: QQ(1), 1: QQ(2)}, 1: {0: QQ(3), 1: QQ(4)}}, (2, 2), QQ)
    assert A.rref() == (eye2, [0, 1])
    A = SDM({0: {0: QQ(1)}, 1: {0: QQ(3), 1: QQ(4)}}, (2, 2), QQ)
    assert A.rref() == (eye2, [0, 1])
    A = SDM({0: {1: QQ(2)}, 1: {0: QQ(3), 1: QQ(4)}}, (2, 2), QQ)
    assert A.rref() == (eye2, [0, 1])
    A = SDM({0: {0: QQ(1), 1: QQ(2), 2: QQ(3)}, 1: {0: QQ(4), 1: QQ(5), 2: QQ(6)}, 2: {0: QQ(7), 1: QQ(8), 2: QQ(9)}}, (3, 3), QQ)
    Arref = SDM({0: {0: QQ(1), 2: QQ(-1)}, 1: {1: QQ(1), 2: QQ(2)}}, (3, 3), QQ)
    assert A.rref() == (Arref, [0, 1])
    A = SDM({0: {0: QQ(1), 1: QQ(2), 3: QQ(1)}, 1: {0: QQ(1), 1: QQ(1), 2: QQ(9)}}, (2, 4), QQ)
    Arref = SDM({0: {0: QQ(1), 2: QQ(18), 3: QQ(-1)}, 1: {1: QQ(1), 2: QQ(-9), 3: QQ(1)}}, (2, 4), QQ)
    assert A.rref() == (Arref, [0, 1])
    A = SDM({0: {0: QQ(1), 1: QQ(1), 2: QQ(1)}, 1: {0: QQ(1), 1: QQ(2), 2: QQ(2)}}, (2, 3), QQ)
    Arref = SDM({0: {0: QQ(1, 1)}, 1: {1: QQ(1, 1), 2: QQ(1, 1)}}, (2, 3), QQ)
    assert A.rref() == (Arref, [0, 1])