from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polyclasses import DMP, DMF, ANP
from sympy.polys.polyerrors import (CoercionFailed, ExactQuotientFailed,
from sympy.polys.specialpolys import f_polys
from sympy.testing.pytest import raises
def test_DMP_properties():
    assert DMP([[]], ZZ).is_zero is True
    assert DMP([[1]], ZZ).is_zero is False
    assert DMP([[1]], ZZ).is_one is True
    assert DMP([[2]], ZZ).is_one is False
    assert DMP([[1]], ZZ).is_ground is True
    assert DMP([[1], [2], [1]], ZZ).is_ground is False
    assert DMP([[1], [2, 0], [1, 0]], ZZ).is_sqf is True
    assert DMP([[1], [2, 0], [1, 0, 0]], ZZ).is_sqf is False
    assert DMP([[1, 2], [3]], ZZ).is_monic is True
    assert DMP([[2, 2], [3]], ZZ).is_monic is False
    assert DMP([[1, 2], [3]], ZZ).is_primitive is True
    assert DMP([[2, 4], [6]], ZZ).is_primitive is False