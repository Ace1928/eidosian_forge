from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
from sympy.polys import polyconfig as config
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises
def test_gf_gcd():
    assert gf_gcd(ZZ.map([]), ZZ.map([]), 11, ZZ) == []
    assert gf_gcd(ZZ.map([2]), ZZ.map([]), 11, ZZ) == [1]
    assert gf_gcd(ZZ.map([]), ZZ.map([2]), 11, ZZ) == [1]
    assert gf_gcd(ZZ.map([2]), ZZ.map([2]), 11, ZZ) == [1]
    assert gf_gcd(ZZ.map([]), ZZ.map([1, 0]), 11, ZZ) == [1, 0]
    assert gf_gcd(ZZ.map([1, 0]), ZZ.map([]), 11, ZZ) == [1, 0]
    assert gf_gcd(ZZ.map([3, 0]), ZZ.map([3, 0]), 11, ZZ) == [1, 0]
    assert gf_gcd(ZZ.map([1, 8, 7]), ZZ.map([1, 7, 1, 7]), 11, ZZ) == [1, 7]