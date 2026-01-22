from sympy.core.numbers import (AlgebraicNumber, E, Float, I, Integer,
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.polys.polytools import Poly
from sympy.abc import x, y, z
from sympy.external.gmpy import HAS_GMPY
from sympy.polys.domains import (ZZ, QQ, RR, CC, FF, GF, EX, EXRAW, ZZ_gmpy,
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.gaussiandomains import ZZ_I, QQ_I
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.domains.realfield import RealField
from sympy.polys.numberfields.subfield import field_isomorphism
from sympy.polys.rings import ring
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.polys.fields import field
from sympy.polys.agca.extensions import FiniteExtension
from sympy.polys.polyerrors import (
from sympy.testing.pytest import raises
from itertools import product
def test_Domain_is_unit():
    nums = [-2, -1, 0, 1, 2]
    invring = [False, True, False, True, False]
    invfield = [True, True, False, True, True]
    ZZx, QQx, QQxf = (ZZ[x], QQ[x], QQ.frac_field(x))
    assert [ZZ.is_unit(ZZ(n)) for n in nums] == invring
    assert [QQ.is_unit(QQ(n)) for n in nums] == invfield
    assert [ZZx.is_unit(ZZx(n)) for n in nums] == invring
    assert [QQx.is_unit(QQx(n)) for n in nums] == invfield
    assert [QQxf.is_unit(QQxf(n)) for n in nums] == invfield
    assert ZZx.is_unit(ZZx(x)) is False
    assert QQx.is_unit(QQx(x)) is False
    assert QQxf.is_unit(QQxf(x)) is True