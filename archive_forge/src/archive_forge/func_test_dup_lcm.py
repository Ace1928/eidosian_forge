from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, RR
from sympy.polys.specialpolys import (
def test_dup_lcm():
    R, x = ring('x', ZZ)
    assert R.dup_lcm(2, 6) == 6
    assert R.dup_lcm(2 * x ** 3, 6 * x) == 6 * x ** 3
    assert R.dup_lcm(2 * x ** 3, 3 * x) == 6 * x ** 3
    assert R.dup_lcm(x ** 2 + x, x) == x ** 2 + x
    assert R.dup_lcm(x ** 2 + x, 2 * x) == 2 * x ** 2 + 2 * x
    assert R.dup_lcm(x ** 2 + 2 * x, x) == x ** 2 + 2 * x
    assert R.dup_lcm(2 * x ** 2 + x, x) == 2 * x ** 2 + x
    assert R.dup_lcm(2 * x ** 2 + x, 2 * x) == 4 * x ** 2 + 2 * x