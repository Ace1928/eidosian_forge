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
def test_drop():
    assert ZZ.drop(x) == ZZ
    assert ZZ[x].drop(x) == ZZ
    assert ZZ[x, y].drop(x) == ZZ[y]
    assert ZZ.frac_field(x).drop(x) == ZZ
    assert ZZ.frac_field(x, y).drop(x) == ZZ.frac_field(y)
    assert ZZ[x][y].drop(y) == ZZ[x]
    assert ZZ[x][y].drop(x) == ZZ[y]
    assert ZZ.frac_field(x)[y].drop(x) == ZZ[y]
    assert ZZ.frac_field(x)[y].drop(y) == ZZ.frac_field(x)
    Ky = FiniteExtension(Poly(x ** 2 - 1, x, domain=ZZ[y]))
    K = FiniteExtension(Poly(x ** 2 - 1, x, domain=ZZ))
    assert Ky.drop(y) == K
    raises(GeneratorsError, lambda: Ky.drop(x))