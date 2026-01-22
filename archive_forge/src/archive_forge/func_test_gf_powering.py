from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
from sympy.polys import polyconfig as config
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises
def test_gf_powering():
    assert gf_pow([1, 0, 0, 1, 8], 0, 11, ZZ) == [1]
    assert gf_pow([1, 0, 0, 1, 8], 1, 11, ZZ) == [1, 0, 0, 1, 8]
    assert gf_pow([1, 0, 0, 1, 8], 2, 11, ZZ) == [1, 0, 0, 2, 5, 0, 1, 5, 9]
    assert gf_pow([1, 0, 0, 1, 8], 5, 11, ZZ) == [1, 0, 0, 5, 7, 0, 10, 6, 2, 10, 9, 6, 10, 6, 6, 0, 5, 2, 5, 9, 10]
    assert gf_pow([1, 0, 0, 1, 8], 8, 11, ZZ) == [1, 0, 0, 8, 9, 0, 6, 8, 10, 1, 2, 5, 10, 7, 7, 9, 1, 2, 0, 0, 6, 2, 5, 2, 5, 7, 7, 9, 10, 10, 7, 5, 5]
    assert gf_pow([1, 0, 0, 1, 8], 45, 11, ZZ) == [1, 0, 0, 1, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 10, 0, 0, 0, 0, 0, 0, 10, 0, 0, 10, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 6, 4, 0, 0, 0, 0, 0, 0, 8, 0, 0, 8, 9, 0, 0, 0, 0, 0, 0, 10, 0, 0, 10, 3, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 10, 0, 0, 0, 0, 0, 0, 8, 0, 0, 8, 9, 0, 0, 0, 0, 0, 0, 9, 0, 0, 9, 6, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 10, 0, 0, 10, 3, 0, 0, 0, 0, 0, 0, 10, 0, 0, 10, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 5, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 10]
    assert gf_pow_mod(ZZ.map([1, 0, 0, 1, 8]), 0, ZZ.map([2, 0, 7]), 11, ZZ) == [1]
    assert gf_pow_mod(ZZ.map([1, 0, 0, 1, 8]), 1, ZZ.map([2, 0, 7]), 11, ZZ) == [1, 1]
    assert gf_pow_mod(ZZ.map([1, 0, 0, 1, 8]), 2, ZZ.map([2, 0, 7]), 11, ZZ) == [2, 3]
    assert gf_pow_mod(ZZ.map([1, 0, 0, 1, 8]), 5, ZZ.map([2, 0, 7]), 11, ZZ) == [7, 8]
    assert gf_pow_mod(ZZ.map([1, 0, 0, 1, 8]), 8, ZZ.map([2, 0, 7]), 11, ZZ) == [1, 5]
    assert gf_pow_mod(ZZ.map([1, 0, 0, 1, 8]), 45, ZZ.map([2, 0, 7]), 11, ZZ) == [5, 4]