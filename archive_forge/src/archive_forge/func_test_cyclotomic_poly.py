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
def test_cyclotomic_poly():
    raises(ValueError, lambda: cyclotomic_poly(0, x))
    assert cyclotomic_poly(1, x, polys=True) == Poly(x - 1)
    assert cyclotomic_poly(1, x) == x - 1
    assert cyclotomic_poly(2, x) == x + 1
    assert cyclotomic_poly(3, x) == x ** 2 + x + 1
    assert cyclotomic_poly(4, x) == x ** 2 + 1
    assert cyclotomic_poly(5, x) == x ** 4 + x ** 3 + x ** 2 + x + 1
    assert cyclotomic_poly(6, x) == x ** 2 - x + 1