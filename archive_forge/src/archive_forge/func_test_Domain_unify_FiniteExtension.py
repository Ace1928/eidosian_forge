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
def test_Domain_unify_FiniteExtension():
    KxZZ = FiniteExtension(Poly(x ** 2 - 2, x, domain=ZZ))
    KxQQ = FiniteExtension(Poly(x ** 2 - 2, x, domain=QQ))
    KxZZy = FiniteExtension(Poly(x ** 2 - 2, x, domain=ZZ[y]))
    KxQQy = FiniteExtension(Poly(x ** 2 - 2, x, domain=QQ[y]))
    assert KxZZ.unify(KxZZ) == KxZZ
    assert KxQQ.unify(KxQQ) == KxQQ
    assert KxZZy.unify(KxZZy) == KxZZy
    assert KxQQy.unify(KxQQy) == KxQQy
    assert KxZZ.unify(ZZ) == KxZZ
    assert KxZZ.unify(QQ) == KxQQ
    assert KxQQ.unify(ZZ) == KxQQ
    assert KxQQ.unify(QQ) == KxQQ
    assert KxZZ.unify(ZZ[y]) == KxZZy
    assert KxZZ.unify(QQ[y]) == KxQQy
    assert KxQQ.unify(ZZ[y]) == KxQQy
    assert KxQQ.unify(QQ[y]) == KxQQy
    assert KxZZy.unify(ZZ) == KxZZy
    assert KxZZy.unify(QQ) == KxQQy
    assert KxQQy.unify(ZZ) == KxQQy
    assert KxQQy.unify(QQ) == KxQQy
    assert KxZZy.unify(ZZ[y]) == KxZZy
    assert KxZZy.unify(QQ[y]) == KxQQy
    assert KxQQy.unify(ZZ[y]) == KxQQy
    assert KxQQy.unify(QQ[y]) == KxQQy
    K = FiniteExtension(Poly(x ** 2 - 2, x, domain=ZZ[y]))
    assert K.unify(ZZ) == K
    assert K.unify(ZZ[x]) == K
    assert K.unify(ZZ[y]) == K
    assert K.unify(ZZ[x, y]) == K
    Kz = FiniteExtension(Poly(x ** 2 - 2, x, domain=ZZ[y, z]))
    assert K.unify(ZZ[z]) == Kz
    assert K.unify(ZZ[x, z]) == Kz
    assert K.unify(ZZ[y, z]) == Kz
    assert K.unify(ZZ[x, y, z]) == Kz
    Kx = FiniteExtension(Poly(x ** 2 - 2, x, domain=ZZ))
    Ky = FiniteExtension(Poly(y ** 2 - 2, y, domain=ZZ))
    Kxy = FiniteExtension(Poly(y ** 2 - 2, y, domain=Kx))
    assert Kx.unify(Kx) == Kx
    assert Ky.unify(Ky) == Ky
    assert Kx.unify(Ky) == Kxy
    assert Ky.unify(Kx) == Kxy