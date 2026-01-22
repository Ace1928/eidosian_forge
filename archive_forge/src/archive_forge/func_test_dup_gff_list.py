from sympy.polys.rings import ring
from sympy.polys.domains import FF, ZZ, QQ
from sympy.polys.specialpolys import f_polys
from sympy.testing.pytest import raises
def test_dup_gff_list():
    R, x = ring('x', ZZ)
    f = x ** 5 + 2 * x ** 4 - x ** 3 - 2 * x ** 2
    assert R.dup_gff_list(f) == [(x, 1), (x + 2, 4)]
    g = x ** 9 - 20 * x ** 8 + 166 * x ** 7 - 744 * x ** 6 + 1965 * x ** 5 - 3132 * x ** 4 + 2948 * x ** 3 - 1504 * x ** 2 + 320 * x
    assert R.dup_gff_list(g) == [(x ** 2 - 5 * x + 4, 1), (x ** 2 - 5 * x + 4, 2), (x, 3)]
    raises(ValueError, lambda: R.dup_gff_list(0))