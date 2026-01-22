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
def test_swinnerton_dyer_poly():
    raises(ValueError, lambda: swinnerton_dyer_poly(0, x))
    assert swinnerton_dyer_poly(1, x, polys=True) == Poly(x ** 2 - 2)
    assert swinnerton_dyer_poly(1, x) == x ** 2 - 2
    assert swinnerton_dyer_poly(2, x) == x ** 4 - 10 * x ** 2 + 1
    assert swinnerton_dyer_poly(3, x) == x ** 8 - 40 * x ** 6 + 352 * x ** 4 - 960 * x ** 2 + 576
    p = [sqrt(prime(i)) for i in range(1, 5)]
    assert str([i.n(3) for i in swinnerton_dyer_poly(4, polys=True).all_roots()]) == str(sorted([Add(*i).n(3) for i in permute_signs(p)]))