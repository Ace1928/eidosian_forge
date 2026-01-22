from sympy.abc import x
from sympy.combinatorics.galois import (
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.numberfields.galoisgroups import (
from sympy.polys.numberfields.subfield import field_isomorphism
from sympy.polys.polytools import Poly
from sympy.testing.pytest import raises
def test_tschirnhausen_transformation():
    for T in [Poly(x ** 2 - 2), Poly(x ** 2 + x + 1), Poly(x ** 4 + 1), Poly(x ** 4 - x ** 3 + x ** 2 - x + 1)]:
        _, U = tschirnhausen_transformation(T)
        assert U.degree() == T.degree()
        assert U.is_monic
        assert U.is_irreducible
        K = QQ.alg_field_from_poly(T)
        L = QQ.alg_field_from_poly(U)
        assert field_isomorphism(K.ext, L.ext) is not None