from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
from sympy.polys import polyconfig as config
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises
def test_gf_expand():
    F = [([1, 1], 2), ([1, 2], 3)]
    assert gf_expand(F, 11, ZZ) == [1, 8, 3, 5, 6, 8]
    assert gf_expand((4, F), 11, ZZ) == [4, 10, 1, 9, 2, 10]