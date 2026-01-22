from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
from sympy.polys import polyconfig as config
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises
def test_gf_berlekamp():
    f = gf_from_int_poly([1, -3, 1, -3, -1, -3, 1], 11)
    Q = [[1, 0, 0, 0, 0, 0], [3, 5, 8, 8, 6, 5], [3, 6, 6, 1, 10, 0], [9, 4, 10, 3, 7, 9], [7, 8, 10, 0, 0, 8], [8, 10, 7, 8, 10, 8]]
    V = [[1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0], [0, 0, 7, 9, 0, 1]]
    assert gf_Qmatrix(f, 11, ZZ) == Q
    assert gf_Qbasis(Q, 11, ZZ) == V
    assert gf_berlekamp(f, 11, ZZ) == [[1, 1], [1, 5, 3], [1, 2, 3, 4]]
    f = ZZ.map([1, 0, 1, 0, 10, 10, 8, 2, 8])
    Q = ZZ.map([[1, 0, 0, 0, 0, 0, 0, 0], [2, 1, 7, 11, 10, 12, 5, 11], [3, 6, 4, 3, 0, 4, 7, 2], [4, 3, 6, 5, 1, 6, 2, 3], [2, 11, 8, 8, 3, 1, 3, 11], [6, 11, 8, 6, 2, 7, 10, 9], [5, 11, 7, 10, 0, 11, 7, 12], [3, 3, 12, 5, 0, 11, 9, 12]])
    V = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 5, 5, 0, 9, 5, 1, 0], [0, 9, 11, 9, 10, 12, 0, 1]]
    assert gf_Qmatrix(f, 13, ZZ) == Q
    assert gf_Qbasis(Q, 13, ZZ) == V
    assert gf_berlekamp(f, 13, ZZ) == [[1, 3], [1, 8, 4, 12], [1, 2, 3, 4, 6]]