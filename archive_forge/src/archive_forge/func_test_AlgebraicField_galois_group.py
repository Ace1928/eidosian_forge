from sympy.abc import x
from sympy.combinatorics.galois import (
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.numberfields.galoisgroups import (
from sympy.polys.numberfields.subfield import field_isomorphism
from sympy.polys.polytools import Poly
from sympy.testing.pytest import raises
def test_AlgebraicField_galois_group():
    k = QQ.alg_field_from_poly(Poly(x ** 4 + 1))
    G, _ = k.galois_group(by_name=True)
    assert G == S4TransitiveSubgroups.V
    k = QQ.alg_field_from_poly(Poly(x ** 4 - 2))
    G, _ = k.galois_group(by_name=True)
    assert G == S4TransitiveSubgroups.D4