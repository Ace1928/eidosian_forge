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
def test_DomainMatrix_diag():
    A = DomainMatrix({0: {0: ZZ(2)}, 1: {1: ZZ(3)}}, (2, 2), ZZ)
    assert DomainMatrix.diag([ZZ(2), ZZ(3)], ZZ) == A
    A = DomainMatrix({0: {0: ZZ(2)}, 1: {1: ZZ(3)}}, (3, 4), ZZ)
    assert DomainMatrix.diag([ZZ(2), ZZ(3)], ZZ, (3, 4)) == A