from sympy.polys.groebnertools import (
from sympy.polys.fglmtools import _representing_matrices
from sympy.polys.orderings import lex, grlex
from sympy.polys.rings import ring, xring
from sympy.polys.domains import ZZ, QQ
from sympy.testing.pytest import slow
from sympy.polys import polyconfig as config
def test_is_rewritable_or_comparable():
    R, x, y, z, t = ring('x,y,z,t', QQ, grlex)
    p = lbp(sig((0, 0, 2, 1), 2), R.zero, 2)
    B = [lbp(sig((0, 0, 0, 1), 2), QQ(2, 45) * y ** 2 + QQ(1, 5) * y * z + QQ(5, 63) * y * t + z ** 2 * t + QQ(4, 45) * z ** 2 + QQ(76, 35) * z * t ** 2 - QQ(32, 105) * z * t + QQ(13, 7) * t ** 3 - QQ(13, 21) * t ** 2, 6)]
    assert is_rewritable_or_comparable(Sign(p), Num(p), B) is True
    p = lbp(sig((0, 1, 1, 0), 2), R.zero, 7)
    B = [lbp(sig((0, 0, 0, 0), 3), QQ(10, 3) * y * z + QQ(4, 3) * y * t - QQ(1, 3) * y + 4 * z ** 2 + QQ(22, 3) * z * t - QQ(4, 3) * z + 4 * t ** 2 - QQ(4, 3) * t, 3)]
    assert is_rewritable_or_comparable(Sign(p), Num(p), B) is True