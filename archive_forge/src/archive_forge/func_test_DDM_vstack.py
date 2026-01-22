from sympy.testing.pytest import raises
from sympy.external.gmpy import HAS_GMPY
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (
def test_DDM_vstack():
    A = DDM([[ZZ(1)], [ZZ(2)], [ZZ(3)]], (3, 1), ZZ)
    B = DDM([[ZZ(4)], [ZZ(5)]], (2, 1), ZZ)
    C = DDM([[ZZ(6)]], (1, 1), ZZ)
    Ah = A.vstack(B)
    assert Ah.shape == (5, 1)
    assert Ah.domain == ZZ
    assert Ah == DDM([[ZZ(1)], [ZZ(2)], [ZZ(3)], [ZZ(4)], [ZZ(5)]], (5, 1), ZZ)
    Ah = A.vstack(B, C)
    assert Ah.shape == (6, 1)
    assert Ah.domain == ZZ
    assert Ah == DDM([[ZZ(1)], [ZZ(2)], [ZZ(3)], [ZZ(4)], [ZZ(5)], [ZZ(6)]], (6, 1), ZZ)