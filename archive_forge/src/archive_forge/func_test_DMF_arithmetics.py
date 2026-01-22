from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polyclasses import DMP, DMF, ANP
from sympy.polys.polyerrors import (CoercionFailed, ExactQuotientFailed,
from sympy.polys.specialpolys import f_polys
from sympy.testing.pytest import raises
def test_DMF_arithmetics():
    f = DMF([[7], [-9]], ZZ)
    g = DMF([[-7], [9]], ZZ)
    assert f.neg() == -f == g
    f = DMF(([[1]], [[1], []]), ZZ)
    g = DMF(([[1]], [[1, 0]]), ZZ)
    h = DMF(([[1], [1, 0]], [[1, 0], []]), ZZ)
    assert f.add(g) == f + g == h
    assert g.add(f) == g + f == h
    h = DMF(([[-1], [1, 0]], [[1, 0], []]), ZZ)
    assert f.sub(g) == f - g == h
    h = DMF(([[1]], [[1, 0], []]), ZZ)
    assert f.mul(g) == f * g == h
    assert g.mul(f) == g * f == h
    h = DMF(([[1, 0]], [[1], []]), ZZ)
    assert f.quo(g) == f / g == h
    h = DMF(([[1]], [[1], [], [], []]), ZZ)
    assert f.pow(3) == f ** 3 == h
    h = DMF(([[1]], [[1, 0, 0, 0]]), ZZ)
    assert g.pow(3) == g ** 3 == h
    h = DMF(([[1, 0]], [[1]]), ZZ)
    assert g.pow(-1) == g ** (-1) == h