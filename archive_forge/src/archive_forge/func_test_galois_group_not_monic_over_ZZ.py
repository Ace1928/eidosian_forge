from sympy.abc import x
from sympy.combinatorics.galois import (
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.numberfields.galoisgroups import (
from sympy.polys.numberfields.subfield import field_isomorphism
from sympy.polys.polytools import Poly
from sympy.testing.pytest import raises
def test_galois_group_not_monic_over_ZZ():
    """
    Check that we can work with polys that are not monic over ZZ.
    """
    for deg in range(1, 7):
        T, G, alt = test_polys_by_deg[deg][0]
        assert galois_group(T / 2, by_name=True) == (G, alt)