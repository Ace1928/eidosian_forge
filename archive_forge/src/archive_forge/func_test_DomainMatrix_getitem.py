from sympy.testing.pytest import raises
from sympy.core.numbers import Integer, Rational
from sympy.core.singleton import S
from sympy.functions import sqrt
from sympy.matrices.dense import Matrix
from sympy.polys.domains import FF, ZZ, QQ, EXRAW
from sympy.polys.matrices.domainmatrix import DomainMatrix, DomainScalar, DM
from sympy.polys.matrices.exceptions import (
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.sdm import SDM
def test_DomainMatrix_getitem():
    dM = DomainMatrix([[ZZ(1), ZZ(2), ZZ(3)], [ZZ(4), ZZ(5), ZZ(6)], [ZZ(7), ZZ(8), ZZ(9)]], (3, 3), ZZ)
    assert dM[1:, :-2] == DomainMatrix([[ZZ(4)], [ZZ(7)]], (2, 1), ZZ)
    assert dM[2, :-2] == DomainMatrix([[ZZ(7)]], (1, 1), ZZ)
    assert dM[:-2, :-2] == DomainMatrix([[ZZ(1)]], (1, 1), ZZ)
    assert dM[:-1, 0:2] == DomainMatrix([[ZZ(1), ZZ(2)], [ZZ(4), ZZ(5)]], (2, 2), ZZ)
    assert dM[:, -1] == DomainMatrix([[ZZ(3)], [ZZ(6)], [ZZ(9)]], (3, 1), ZZ)
    assert dM[-1, :] == DomainMatrix([[ZZ(7), ZZ(8), ZZ(9)]], (1, 3), ZZ)
    assert dM[::-1, :] == DomainMatrix([[ZZ(7), ZZ(8), ZZ(9)], [ZZ(4), ZZ(5), ZZ(6)], [ZZ(1), ZZ(2), ZZ(3)]], (3, 3), ZZ)
    raises(IndexError, lambda: dM[4, :-2])
    raises(IndexError, lambda: dM[:-2, 4])
    assert dM[1, 2] == DomainScalar(ZZ(6), ZZ)
    assert dM[-2, 2] == DomainScalar(ZZ(6), ZZ)
    assert dM[1, -2] == DomainScalar(ZZ(5), ZZ)
    assert dM[-1, -3] == DomainScalar(ZZ(7), ZZ)
    raises(IndexError, lambda: dM[3, 3])
    raises(IndexError, lambda: dM[1, 4])
    raises(IndexError, lambda: dM[-1, -4])
    dM = DomainMatrix({0: {0: ZZ(1)}}, (10, 10), ZZ)
    assert dM[5, 5] == DomainScalar(ZZ(0), ZZ)
    assert dM[0, 0] == DomainScalar(ZZ(1), ZZ)
    dM = DomainMatrix({1: {0: 1}}, (2, 1), ZZ)
    assert dM[0:, 0] == DomainMatrix({1: {0: 1}}, (2, 1), ZZ)
    raises(IndexError, lambda: dM[3, 0])
    dM = DomainMatrix({2: {2: ZZ(1)}, 4: {4: ZZ(1)}}, (5, 5), ZZ)
    assert dM[:2, :2] == DomainMatrix({}, (2, 2), ZZ)
    assert dM[2:, 2:] == DomainMatrix({0: {0: 1}, 2: {2: 1}}, (3, 3), ZZ)
    assert dM[3:, 3:] == DomainMatrix({1: {1: 1}}, (2, 2), ZZ)
    assert dM[2:, 6:] == DomainMatrix({}, (3, 0), ZZ)