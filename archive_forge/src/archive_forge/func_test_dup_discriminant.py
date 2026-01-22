from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, RR
from sympy.polys.specialpolys import (
def test_dup_discriminant():
    R, x = ring('x', ZZ)
    assert R.dup_discriminant(0) == 0
    assert R.dup_discriminant(x) == 1
    assert R.dup_discriminant(x ** 3 + 3 * x ** 2 + 9 * x - 13) == -11664
    assert R.dup_discriminant(5 * x ** 5 + x ** 3 + 2) == 31252160
    assert R.dup_discriminant(x ** 4 + 2 * x ** 3 + 6 * x ** 2 - 22 * x + 13) == 0
    assert R.dup_discriminant(12 * x ** 7 + 15 * x ** 4 + 30 * x ** 3 + x ** 2 + 1) == -220289699947514112