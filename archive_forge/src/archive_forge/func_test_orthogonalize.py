from sympy.vector.vector import Vector
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.functions import express, matrix_to_vector, orthogonalize
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.testing.pytest import raises
def test_orthogonalize():
    C = CoordSys3D('C')
    a, b = symbols('a b', integer=True)
    i, j, k = C.base_vectors()
    v1 = i + 2 * j
    v2 = 2 * i + 3 * j
    v3 = 3 * i + 5 * j
    v4 = 3 * i + j
    v5 = 2 * i + 2 * j
    v6 = a * i + b * j
    v7 = 4 * a * i + 4 * b * j
    assert orthogonalize(v1, v2) == [C.i + 2 * C.j, C.i * Rational(2, 5) + -C.j / 5]
    assert orthogonalize(v4, v5, orthonormal=True) == [3 * sqrt(10) * C.i / 10 + sqrt(10) * C.j / 10, -sqrt(10) * C.i / 10 + 3 * sqrt(10) * C.j / 10]
    raises(ValueError, lambda: orthogonalize(v1, v2, v3))
    raises(ValueError, lambda: orthogonalize(v6, v7))