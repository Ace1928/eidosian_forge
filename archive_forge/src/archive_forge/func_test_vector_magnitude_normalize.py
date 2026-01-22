from sympy.core import Rational, S
from sympy.simplify import simplify, trigsimp
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.vector.vector import Vector, BaseVector, VectorAdd, \
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.vector import Cross, Dot, cross
from sympy.testing.pytest import raises
def test_vector_magnitude_normalize():
    assert Vector.zero.magnitude() == 0
    assert Vector.zero.normalize() == Vector.zero
    assert i.magnitude() == 1
    assert j.magnitude() == 1
    assert k.magnitude() == 1
    assert i.normalize() == i
    assert j.normalize() == j
    assert k.normalize() == k
    v1 = a * i
    assert v1.normalize() == a / sqrt(a ** 2) * i
    assert v1.magnitude() == sqrt(a ** 2)
    v2 = a * i + b * j + c * k
    assert v2.magnitude() == sqrt(a ** 2 + b ** 2 + c ** 2)
    assert v2.normalize() == v2 / v2.magnitude()
    v3 = i + j
    assert v3.normalize() == sqrt(2) / 2 * C.i + sqrt(2) / 2 * C.j