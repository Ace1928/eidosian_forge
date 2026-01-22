from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
from sympy.polys import polyconfig as config
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises
def test_gf_from_to_dict():
    f = {11: 12, 6: 2, 0: 25}
    F = {11: 1, 6: 2, 0: 3}
    g = [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3]
    assert gf_from_dict(f, 11, ZZ) == g
    assert gf_to_dict(g, 11) == F
    f = {11: -5, 4: 0, 3: 1, 0: 12}
    F = {11: -5, 3: 1, 0: 1}
    g = [6, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
    assert gf_from_dict(f, 11, ZZ) == g
    assert gf_to_dict(g, 11) == F
    assert gf_to_dict([10], 11, symmetric=True) == {0: -1}
    assert gf_to_dict([10], 11, symmetric=False) == {0: 10}