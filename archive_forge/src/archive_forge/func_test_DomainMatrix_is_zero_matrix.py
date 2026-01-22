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
def test_DomainMatrix_is_zero_matrix():
    A = DomainMatrix([[ZZ(1)]], (1, 1), ZZ)
    B = DomainMatrix([[ZZ(0)]], (1, 1), ZZ)
    assert A.is_zero_matrix is False
    assert B.is_zero_matrix is True