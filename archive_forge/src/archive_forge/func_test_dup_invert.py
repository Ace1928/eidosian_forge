from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, RR
from sympy.polys.specialpolys import (
def test_dup_invert():
    R, x = ring('x', QQ)
    assert R.dup_invert(2 * x, x ** 2 - 16) == QQ(1, 32) * x