from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, RR
from sympy.polys.specialpolys import (
def test_dmp_primitive():
    R, x, y = ring('x,y', ZZ)
    assert R.dmp_primitive(0) == (0, 0)
    assert R.dmp_primitive(1) == (1, 1)
    f, g, F = (3 * y ** 2 + 2 * y + 1, 1, 0)
    for i in range(0, 5):
        g *= f
        F += x ** i * g
    assert R.dmp_primitive(F) == (f.drop(x), F / f)
    R, x, y, z = ring('x,y,z', ZZ)
    cont, f = R.dmp_primitive(f_4)
    assert cont == 1 and f == f_4
    cont, f = R.dmp_primitive(f_5)
    assert cont == 1 and f == f_5
    R, x, y, z, t = ring('x,y,z,t', ZZ)
    cont, f = R.dmp_primitive(f_6)
    assert cont == 1 and f == f_6