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
def test_Domain_convert():

    def check_element(e1, e2, K1, K2, K3):
        assert type(e1) is type(e2), '%s, %s: %s %s -> %s' % (e1, e2, K1, K2, K3)
        assert e1 == e2, '%s, %s: %s %s -> %s' % (e1, e2, K1, K2, K3)

    def check_domains(K1, K2):
        K3 = K1.unify(K2)
        check_element(K3.convert_from(K1.one, K1), K3.one, K1, K2, K3)
        check_element(K3.convert_from(K2.one, K2), K3.one, K1, K2, K3)
        check_element(K3.convert_from(K1.zero, K1), K3.zero, K1, K2, K3)
        check_element(K3.convert_from(K2.zero, K2), K3.zero, K1, K2, K3)

    def composite_domains(K):
        domains = [K, K[y], K[z], K[y, z], K.frac_field(y), K.frac_field(z), K.frac_field(y, z)]
        return domains
    QQ2 = QQ.algebraic_field(sqrt(2))
    QQ3 = QQ.algebraic_field(sqrt(3))
    doms = [ZZ, QQ, QQ2, QQ3, QQ_I, ZZ_I, RR, CC]
    for i, K1 in enumerate(doms):
        for K2 in doms[i:]:
            for K3 in composite_domains(K1):
                for K4 in composite_domains(K2):
                    check_domains(K3, K4)
    assert QQ.convert(1e-51) == QQ(1684996666696915, 1684996666696914987166688442938726917102321526408785780068975640576)
    R, xr = ring('x', ZZ)
    assert ZZ.convert(xr - xr) == 0
    assert ZZ.convert(xr - xr, R.to_domain()) == 0
    assert CC.convert(ZZ_I(1, 2)) == CC(1, 2)
    assert CC.convert(QQ_I(1, 2)) == CC(1, 2)
    K1 = QQ.frac_field(x)
    K2 = ZZ.frac_field(x)
    K3 = QQ[x]
    K4 = ZZ[x]
    Ks = [K1, K2, K3, K4]
    for Ka, Kb in product(Ks, Ks):
        assert Ka.convert_from(Kb.from_sympy(x), Kb) == Ka.from_sympy(x)
    assert K2.convert_from(QQ(1, 2), QQ) == K2(QQ(1, 2))