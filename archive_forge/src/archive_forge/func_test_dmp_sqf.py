from sympy.polys.rings import ring
from sympy.polys.domains import FF, ZZ, QQ
from sympy.polys.specialpolys import f_polys
from sympy.testing.pytest import raises
def test_dmp_sqf():
    R, x, y = ring('x,y', ZZ)
    assert R.dmp_sqf_part(0) == 0
    assert R.dmp_sqf_p(0) is True
    assert R.dmp_sqf_part(7) == 1
    assert R.dmp_sqf_p(7) is True
    assert R.dmp_sqf_list(3) == (3, [])
    assert R.dmp_sqf_list_include(3) == [(3, 1)]
    R, x, y, z = ring('x,y,z', ZZ)
    assert R.dmp_sqf_p(f_0) is True
    assert R.dmp_sqf_p(f_0 ** 2) is False
    assert R.dmp_sqf_p(f_1) is True
    assert R.dmp_sqf_p(f_1 ** 2) is False
    assert R.dmp_sqf_p(f_2) is True
    assert R.dmp_sqf_p(f_2 ** 2) is False
    assert R.dmp_sqf_p(f_3) is True
    assert R.dmp_sqf_p(f_3 ** 2) is False
    assert R.dmp_sqf_p(f_5) is False
    assert R.dmp_sqf_p(f_5 ** 2) is False
    assert R.dmp_sqf_p(f_4) is True
    assert R.dmp_sqf_part(f_4) == -f_4
    assert R.dmp_sqf_part(f_5) == x + y - z
    R, x, y, z, t = ring('x,y,z,t', ZZ)
    assert R.dmp_sqf_p(f_6) is True
    assert R.dmp_sqf_part(f_6) == f_6
    R, x = ring('x', ZZ)
    f = -x ** 5 + x ** 4 + x - 1
    assert R.dmp_sqf_list(f) == (-1, [(x ** 3 + x ** 2 + x + 1, 1), (x - 1, 2)])
    assert R.dmp_sqf_list_include(f) == [(-x ** 3 - x ** 2 - x - 1, 1), (x - 1, 2)]
    R, x, y = ring('x,y', ZZ)
    f = -x ** 5 + x ** 4 + x - 1
    assert R.dmp_sqf_list(f) == (-1, [(x ** 3 + x ** 2 + x + 1, 1), (x - 1, 2)])
    assert R.dmp_sqf_list_include(f) == [(-x ** 3 - x ** 2 - x - 1, 1), (x - 1, 2)]
    f = -x ** 2 + 2 * x - 1
    assert R.dmp_sqf_list_include(f) == [(-1, 1), (x - 1, 2)]
    R, x, y = ring('x,y', FF(2))
    raises(NotImplementedError, lambda: R.dmp_sqf_list(y ** 2 + 1))