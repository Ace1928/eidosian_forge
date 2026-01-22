from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
from sympy.polys import polyconfig as config
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises
def test_gf_squarefree():
    assert gf_sqf_list([], 11, ZZ) == (0, [])
    assert gf_sqf_list([1], 11, ZZ) == (1, [])
    assert gf_sqf_list([1, 1], 11, ZZ) == (1, [([1, 1], 1)])
    assert gf_sqf_p([], 11, ZZ) is True
    assert gf_sqf_p([1], 11, ZZ) is True
    assert gf_sqf_p([1, 1], 11, ZZ) is True
    f = gf_from_dict({11: 1, 0: 1}, 11, ZZ)
    assert gf_sqf_p(f, 11, ZZ) is False
    assert gf_sqf_list(f, 11, ZZ) == (1, [([1, 1], 11)])
    f = [1, 5, 8, 4]
    assert gf_sqf_p(f, 11, ZZ) is False
    assert gf_sqf_list(f, 11, ZZ) == (1, [([1, 1], 1), ([1, 2], 2)])
    assert gf_sqf_part(f, 11, ZZ) == [1, 3, 2]
    f = [1, 0, 0, 2, 0, 0, 2, 0, 0, 1, 0]
    assert gf_sqf_list(f, 3, ZZ) == (1, [([1, 0], 1), ([1, 1], 3), ([1, 2], 6)])