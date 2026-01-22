from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.physics.vector import Dyadic, Point, ReferenceFrame, Vector
from sympy.physics.vector.functions import (cross, dot, express,
from sympy.testing.pytest import raises
def test_cross_different_frames():
    assert cross(N.x, A.x) == sin(q1) * A.z
    assert cross(N.x, A.y) == cos(q1) * A.z
    assert cross(N.x, A.z) == -sin(q1) * A.x - cos(q1) * A.y
    assert cross(N.y, A.x) == -cos(q1) * A.z
    assert cross(N.y, A.y) == sin(q1) * A.z
    assert cross(N.y, A.z) == cos(q1) * A.x - sin(q1) * A.y
    assert cross(N.z, A.x) == A.y
    assert cross(N.z, A.y) == -A.x
    assert cross(N.z, A.z) == 0
    assert cross(N.x, A.x) == sin(q1) * A.z
    assert cross(N.x, A.y) == cos(q1) * A.z
    assert cross(N.x, A.x + A.y) == sin(q1) * A.z + cos(q1) * A.z
    assert cross(A.x + A.y, N.x) == -sin(q1) * A.z - cos(q1) * A.z
    assert cross(A.x, C.x) == sin(q3) * C.y
    assert cross(A.x, C.y) == -sin(q3) * C.x + cos(q3) * C.z
    assert cross(A.x, C.z) == -cos(q3) * C.y
    assert cross(C.x, A.x) == -sin(q3) * C.y
    assert cross(C.y, A.x) == sin(q3) * C.x - cos(q3) * C.z
    assert cross(C.z, A.x) == cos(q3) * C.y