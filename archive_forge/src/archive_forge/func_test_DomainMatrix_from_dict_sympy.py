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
def test_DomainMatrix_from_dict_sympy():
    sdm = SDM({0: {0: QQ(1, 2)}, 1: {1: QQ(2, 3)}}, (2, 2), QQ)
    sympy_dict = {0: {0: Rational(1, 2)}, 1: {1: Rational(2, 3)}}
    A = DomainMatrix.from_dict_sympy(2, 2, sympy_dict)
    assert A.rep == sdm
    assert A.shape == (2, 2)
    assert A.domain == QQ
    fds = DomainMatrix.from_dict_sympy
    raises(DMBadInputError, lambda: fds(2, 2, {3: {0: Rational(1, 2)}}))
    raises(DMBadInputError, lambda: fds(2, 2, {0: {3: Rational(1, 2)}}))