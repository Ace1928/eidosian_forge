from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
from sympy.polys import polyconfig as config
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises
def test_gf_trunc():
    assert gf_trunc([], 11) == []
    assert gf_trunc([1], 11) == [1]
    assert gf_trunc([22], 11) == []
    assert gf_trunc([12], 11) == [1]
    assert gf_trunc([11, 22, 17, 1, 0], 11) == [6, 1, 0]
    assert gf_trunc([12, 23, 17, 1, 0], 11) == [1, 1, 6, 1, 0]