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
def test_canonical_unit():
    for K in [ZZ, QQ, RR]:
        assert K.canonical_unit(K(2)) == K(1)
        assert K.canonical_unit(K(-2)) == K(-1)
    for K in [ZZ_I, QQ_I]:
        i = K.from_sympy(I)
        assert K.canonical_unit(K(2)) == K(1)
        assert K.canonical_unit(K(2) * i) == -i
        assert K.canonical_unit(-K(2)) == K(-1)
        assert K.canonical_unit(-K(2) * i) == i
    K = ZZ[x]
    assert K.canonical_unit(K(x + 1)) == K(1)
    assert K.canonical_unit(K(-x + 1)) == K(-1)
    K = ZZ_I[x]
    assert K.canonical_unit(K.from_sympy(I * x)) == ZZ_I(0, -1)
    K = ZZ_I.frac_field(x, y)
    i = K.from_sympy(I)
    assert i / i == K.one
    assert (K.one + i) / (i - K.one) == -i