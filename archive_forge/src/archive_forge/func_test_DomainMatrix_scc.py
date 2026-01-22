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
def test_DomainMatrix_scc():
    Ad = DomainMatrix([[ZZ(1), ZZ(2), ZZ(3)], [ZZ(0), ZZ(1), ZZ(0)], [ZZ(2), ZZ(0), ZZ(4)]], (3, 3), ZZ)
    As = Ad.to_sparse()
    Addm = Ad.rep
    Asdm = As.rep
    for A in [Ad, As, Addm, Asdm]:
        assert Ad.scc() == [[1], [0, 2]]