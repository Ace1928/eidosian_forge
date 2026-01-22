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
def test_DomainMatrix_mul_elementwise():
    A = DomainMatrix([[ZZ(2), ZZ(2)], [ZZ(0), ZZ(0)]], (2, 2), ZZ)
    B = DomainMatrix([[ZZ(4), ZZ(0)], [ZZ(3), ZZ(0)]], (2, 2), ZZ)
    C = DomainMatrix([[ZZ(8), ZZ(0)], [ZZ(0), ZZ(0)]], (2, 2), ZZ)
    assert A.mul_elementwise(B) == C
    assert B.mul_elementwise(A) == C