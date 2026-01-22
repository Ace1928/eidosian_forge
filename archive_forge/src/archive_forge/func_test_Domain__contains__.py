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
def test_Domain__contains__():
    assert (0 in EX) is True
    assert (0 in ZZ) is True
    assert (0 in QQ) is True
    assert (0 in RR) is True
    assert (0 in CC) is True
    assert (0 in ALG) is True
    assert (0 in ZZ[x, y]) is True
    assert (0 in QQ[x, y]) is True
    assert (0 in RR[x, y]) is True
    assert (-7 in EX) is True
    assert (-7 in ZZ) is True
    assert (-7 in QQ) is True
    assert (-7 in RR) is True
    assert (-7 in CC) is True
    assert (-7 in ALG) is True
    assert (-7 in ZZ[x, y]) is True
    assert (-7 in QQ[x, y]) is True
    assert (-7 in RR[x, y]) is True
    assert (17 in EX) is True
    assert (17 in ZZ) is True
    assert (17 in QQ) is True
    assert (17 in RR) is True
    assert (17 in CC) is True
    assert (17 in ALG) is True
    assert (17 in ZZ[x, y]) is True
    assert (17 in QQ[x, y]) is True
    assert (17 in RR[x, y]) is True
    assert (Rational(-1, 7) in EX) is True
    assert (Rational(-1, 7) in ZZ) is False
    assert (Rational(-1, 7) in QQ) is True
    assert (Rational(-1, 7) in RR) is True
    assert (Rational(-1, 7) in CC) is True
    assert (Rational(-1, 7) in ALG) is True
    assert (Rational(-1, 7) in ZZ[x, y]) is False
    assert (Rational(-1, 7) in QQ[x, y]) is True
    assert (Rational(-1, 7) in RR[x, y]) is True
    assert (Rational(3, 5) in EX) is True
    assert (Rational(3, 5) in ZZ) is False
    assert (Rational(3, 5) in QQ) is True
    assert (Rational(3, 5) in RR) is True
    assert (Rational(3, 5) in CC) is True
    assert (Rational(3, 5) in ALG) is True
    assert (Rational(3, 5) in ZZ[x, y]) is False
    assert (Rational(3, 5) in QQ[x, y]) is True
    assert (Rational(3, 5) in RR[x, y]) is True
    assert (3.0 in EX) is True
    assert (3.0 in ZZ) is True
    assert (3.0 in QQ) is True
    assert (3.0 in RR) is True
    assert (3.0 in CC) is True
    assert (3.0 in ALG) is True
    assert (3.0 in ZZ[x, y]) is True
    assert (3.0 in QQ[x, y]) is True
    assert (3.0 in RR[x, y]) is True
    assert (3.14 in EX) is True
    assert (3.14 in ZZ) is False
    assert (3.14 in QQ) is True
    assert (3.14 in RR) is True
    assert (3.14 in CC) is True
    assert (3.14 in ALG) is True
    assert (3.14 in ZZ[x, y]) is False
    assert (3.14 in QQ[x, y]) is True
    assert (3.14 in RR[x, y]) is True
    assert (oo in ALG) is False
    assert (oo in ZZ[x, y]) is False
    assert (oo in QQ[x, y]) is False
    assert (-oo in ZZ) is False
    assert (-oo in QQ) is False
    assert (-oo in ALG) is False
    assert (-oo in ZZ[x, y]) is False
    assert (-oo in QQ[x, y]) is False
    assert (sqrt(7) in EX) is True
    assert (sqrt(7) in ZZ) is False
    assert (sqrt(7) in QQ) is False
    assert (sqrt(7) in RR) is True
    assert (sqrt(7) in CC) is True
    assert (sqrt(7) in ALG) is False
    assert (sqrt(7) in ZZ[x, y]) is False
    assert (sqrt(7) in QQ[x, y]) is False
    assert (sqrt(7) in RR[x, y]) is True
    assert (2 * sqrt(3) + 1 in EX) is True
    assert (2 * sqrt(3) + 1 in ZZ) is False
    assert (2 * sqrt(3) + 1 in QQ) is False
    assert (2 * sqrt(3) + 1 in RR) is True
    assert (2 * sqrt(3) + 1 in CC) is True
    assert (2 * sqrt(3) + 1 in ALG) is True
    assert (2 * sqrt(3) + 1 in ZZ[x, y]) is False
    assert (2 * sqrt(3) + 1 in QQ[x, y]) is False
    assert (2 * sqrt(3) + 1 in RR[x, y]) is True
    assert (sin(1) in EX) is True
    assert (sin(1) in ZZ) is False
    assert (sin(1) in QQ) is False
    assert (sin(1) in RR) is True
    assert (sin(1) in CC) is True
    assert (sin(1) in ALG) is False
    assert (sin(1) in ZZ[x, y]) is False
    assert (sin(1) in QQ[x, y]) is False
    assert (sin(1) in RR[x, y]) is True
    assert (x ** 2 + 1 in EX) is True
    assert (x ** 2 + 1 in ZZ) is False
    assert (x ** 2 + 1 in QQ) is False
    assert (x ** 2 + 1 in RR) is False
    assert (x ** 2 + 1 in CC) is False
    assert (x ** 2 + 1 in ALG) is False
    assert (x ** 2 + 1 in ZZ[x]) is True
    assert (x ** 2 + 1 in QQ[x]) is True
    assert (x ** 2 + 1 in RR[x]) is True
    assert (x ** 2 + 1 in ZZ[x, y]) is True
    assert (x ** 2 + 1 in QQ[x, y]) is True
    assert (x ** 2 + 1 in RR[x, y]) is True
    assert (x ** 2 + y ** 2 in EX) is True
    assert (x ** 2 + y ** 2 in ZZ) is False
    assert (x ** 2 + y ** 2 in QQ) is False
    assert (x ** 2 + y ** 2 in RR) is False
    assert (x ** 2 + y ** 2 in CC) is False
    assert (x ** 2 + y ** 2 in ALG) is False
    assert (x ** 2 + y ** 2 in ZZ[x]) is False
    assert (x ** 2 + y ** 2 in QQ[x]) is False
    assert (x ** 2 + y ** 2 in RR[x]) is False
    assert (x ** 2 + y ** 2 in ZZ[x, y]) is True
    assert (x ** 2 + y ** 2 in QQ[x, y]) is True
    assert (x ** 2 + y ** 2 in RR[x, y]) is True
    assert (Rational(3, 2) * x / (y + 1) - z in QQ[x, y, z]) is False