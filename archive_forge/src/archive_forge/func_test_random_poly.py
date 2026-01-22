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
def test_random_poly():
    poly = random_poly(x, 10, -100, 100, polys=False)
    assert Poly(poly).degree() == 10
    assert all((-100 <= coeff <= 100 for coeff in Poly(poly).coeffs())) is True
    poly = random_poly(x, 10, -100, 100, polys=True)
    assert poly.degree() == 10
    assert all((-100 <= coeff <= 100 for coeff in poly.coeffs())) is True