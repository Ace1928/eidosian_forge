from sympy.testing.pytest import raises
from sympy.external.gmpy import HAS_GMPY
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (
def test_DDM_extract():
    dm1 = DDM([[ZZ(1), ZZ(2), ZZ(3)], [ZZ(4), ZZ(5), ZZ(6)], [ZZ(7), ZZ(8), ZZ(9)]], (3, 3), ZZ)
    dm2 = DDM([[ZZ(6), ZZ(4)], [ZZ(3), ZZ(1)]], (2, 2), ZZ)
    assert dm1.extract([1, 0], [2, 0]) == dm2
    assert dm1.extract([-2, 0], [-1, 0]) == dm2
    assert dm1.extract([], []) == DDM.zeros((0, 0), ZZ)
    assert dm1.extract([1], []) == DDM.zeros((1, 0), ZZ)
    assert dm1.extract([], [1]) == DDM.zeros((0, 1), ZZ)
    raises(IndexError, lambda: dm2.extract([2], [0]))
    raises(IndexError, lambda: dm2.extract([0], [2]))
    raises(IndexError, lambda: dm2.extract([-3], [0]))
    raises(IndexError, lambda: dm2.extract([0], [-3]))