from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polyclasses import DMP, DMF, ANP
from sympy.polys.polyerrors import (CoercionFailed, ExactQuotientFailed,
from sympy.polys.specialpolys import f_polys
from sympy.testing.pytest import raises
def test_DMP___init__():
    f = DMP([[0], [], [0, 1, 2], [3]], ZZ)
    assert f.rep == [[1, 2], [3]]
    assert f.dom == ZZ
    assert f.lev == 1
    f = DMP([[1, 2], [3]], ZZ, 1)
    assert f.rep == [[1, 2], [3]]
    assert f.dom == ZZ
    assert f.lev == 1
    f = DMP({(1, 1): 1, (0, 0): 2}, ZZ, 1)
    assert f.rep == [[1, 0], [2]]
    assert f.dom == ZZ
    assert f.lev == 1