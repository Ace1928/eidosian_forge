from sympy.polys.densebasic import (
from sympy.polys.specialpolys import f_polys
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.core.singleton import S
from sympy.testing.pytest import raises
from sympy.core.numbers import oo
def test_dup_terms_gcd():
    assert dup_terms_gcd([], ZZ) == (0, [])
    assert dup_terms_gcd([1, 0, 1], ZZ) == (0, [1, 0, 1])
    assert dup_terms_gcd([1, 0, 1, 0], ZZ) == (1, [1, 0, 1])