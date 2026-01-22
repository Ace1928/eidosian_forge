from sympy.testing.pytest import raises
from sympy.polys.polymatrix import PolyMatrix
from sympy.polys import Poly
from sympy.core.singleton import S
from sympy.matrices.dense import Matrix
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.abc import x, y
def test_polymatrix_repr():
    assert repr(PolyMatrix([[1, 2]], x)) == 'PolyMatrix([[1, 2]], ring=QQ[x])'
    assert repr(PolyMatrix(0, 2, [], x)) == 'PolyMatrix(0, 2, [], ring=QQ[x])'