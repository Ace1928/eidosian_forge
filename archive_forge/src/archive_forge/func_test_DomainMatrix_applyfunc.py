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
def test_DomainMatrix_applyfunc():
    A = DomainMatrix([[ZZ(1), ZZ(2)]], (1, 2), ZZ)
    B = DomainMatrix([[ZZ(2), ZZ(4)]], (1, 2), ZZ)
    assert A.applyfunc(lambda x: 2 * x) == B