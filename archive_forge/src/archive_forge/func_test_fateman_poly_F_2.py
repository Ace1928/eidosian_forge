from sympy.core.add import Add
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.ntheory.generate import prime
from sympy.polys.domains.integerring import ZZ
from sympy.polys.polytools import Poly
from sympy.utilities.iterables import permute_signs
from sympy.testing.pytest import raises
from sympy.polys.specialpolys import (
from sympy.abc import x, y, z
def test_fateman_poly_F_2():
    f, g, h = fateman_poly_F_2(1)
    F, G, H = dmp_fateman_poly_F_2(1, ZZ)
    assert [t.rep.rep for t in [f, g, h]] == [F, G, H]
    f, g, h = fateman_poly_F_2(3)
    F, G, H = dmp_fateman_poly_F_2(3, ZZ)
    assert [t.rep.rep for t in [f, g, h]] == [F, G, H]