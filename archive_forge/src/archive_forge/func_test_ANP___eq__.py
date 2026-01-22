from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polyclasses import DMP, DMF, ANP
from sympy.polys.polyerrors import (CoercionFailed, ExactQuotientFailed,
from sympy.polys.specialpolys import f_polys
from sympy.testing.pytest import raises
def test_ANP___eq__():
    a = ANP([QQ(1), QQ(1)], [QQ(1), QQ(0), QQ(1)], QQ)
    b = ANP([QQ(1), QQ(1)], [QQ(1), QQ(0), QQ(2)], QQ)
    assert (a == a) is True
    assert (a != a) is False
    assert (a == b) is False
    assert (a != b) is True
    b = ANP([QQ(1), QQ(2)], [QQ(1), QQ(0), QQ(1)], QQ)
    assert (a == b) is False
    assert (a != b) is True