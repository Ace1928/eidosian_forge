from sympy.testing.pytest import raises
from sympy.polys.polymatrix import PolyMatrix
from sympy.polys import Poly
from sympy.core.singleton import S
from sympy.matrices.dense import Matrix
from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.abc import x, y
def test_polymatrix_from_Matrix():
    assert PolyMatrix.from_Matrix(Matrix([1, 2]), x) == PolyMatrix([1, 2], x, ring=QQ[x])
    assert PolyMatrix.from_Matrix(Matrix([1]), ring=QQ[x]) == PolyMatrix([1], x)
    pmx = PolyMatrix([1, 2], x)
    pmy = PolyMatrix([1, 2], y)
    assert pmx != pmy
    assert pmx.set_gens(y) == pmy