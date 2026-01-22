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
def test_DomainMatrix_from_list_sympy():
    ddm = DDM([[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    A = DomainMatrix.from_list_sympy(2, 2, [[1, 2], [3, 4]])
    assert A.rep == ddm
    assert A.shape == (2, 2)
    assert A.domain == ZZ
    K = QQ.algebraic_field(sqrt(2))
    ddm = DDM([[K.convert(1 + sqrt(2)), K.convert(2 + sqrt(2))], [K.convert(3 + sqrt(2)), K.convert(4 + sqrt(2))]], (2, 2), K)
    A = DomainMatrix.from_list_sympy(2, 2, [[1 + sqrt(2), 2 + sqrt(2)], [3 + sqrt(2), 4 + sqrt(2)]], extension=True)
    assert A.rep == ddm
    assert A.shape == (2, 2)
    assert A.domain == K