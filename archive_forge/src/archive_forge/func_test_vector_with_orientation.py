from sympy.testing.pytest import raises
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.scalar import BaseScalar
from sympy.core.function import expand
from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, atan2, cos, sin)
from sympy.matrices.dense import zeros
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.simplify.simplify import simplify
from sympy.vector.functions import express
from sympy.vector.point import Point
from sympy.vector.vector import Vector
from sympy.vector.orienters import (AxisOrienter, BodyOrienter,
def test_vector_with_orientation():
    """
    Tests the effects of orientation of coordinate systems on
    basic vector operations.
    """
    N = CoordSys3D('N')
    A = N.orient_new_axis('A', q1, N.k)
    B = A.orient_new_axis('B', q2, A.i)
    C = B.orient_new_axis('C', q3, B.j)
    v1 = a * N.i + b * N.j + c * N.k
    assert v1.to_matrix(A) == Matrix([[a * cos(q1) + b * sin(q1)], [-a * sin(q1) + b * cos(q1)], [c]])
    assert N.i.dot(A.i) == cos(q1)
    assert N.i.dot(A.j) == -sin(q1)
    assert N.i.dot(A.k) == 0
    assert N.j.dot(A.i) == sin(q1)
    assert N.j.dot(A.j) == cos(q1)
    assert N.j.dot(A.k) == 0
    assert N.k.dot(A.i) == 0
    assert N.k.dot(A.j) == 0
    assert N.k.dot(A.k) == 1
    assert N.i.dot(A.i + A.j) == -sin(q1) + cos(q1) == (A.i + A.j).dot(N.i)
    assert A.i.dot(C.i) == cos(q3)
    assert A.i.dot(C.j) == 0
    assert A.i.dot(C.k) == sin(q3)
    assert A.j.dot(C.i) == sin(q2) * sin(q3)
    assert A.j.dot(C.j) == cos(q2)
    assert A.j.dot(C.k) == -sin(q2) * cos(q3)
    assert A.k.dot(C.i) == -cos(q2) * sin(q3)
    assert A.k.dot(C.j) == sin(q2)
    assert A.k.dot(C.k) == cos(q2) * cos(q3)
    assert N.i.cross(A.i) == sin(q1) * A.k
    assert N.i.cross(A.j) == cos(q1) * A.k
    assert N.i.cross(A.k) == -sin(q1) * A.i - cos(q1) * A.j
    assert N.j.cross(A.i) == -cos(q1) * A.k
    assert N.j.cross(A.j) == sin(q1) * A.k
    assert N.j.cross(A.k) == cos(q1) * A.i - sin(q1) * A.j
    assert N.k.cross(A.i) == A.j
    assert N.k.cross(A.j) == -A.i
    assert N.k.cross(A.k) == Vector.zero
    assert N.i.cross(A.i) == sin(q1) * A.k
    assert N.i.cross(A.j) == cos(q1) * A.k
    assert N.i.cross(A.i + A.j) == sin(q1) * A.k + cos(q1) * A.k
    assert (A.i + A.j).cross(N.i) == (-sin(q1) - cos(q1)) * N.k
    assert A.i.cross(C.i) == sin(q3) * C.j
    assert A.i.cross(C.j) == -sin(q3) * C.i + cos(q3) * C.k
    assert A.i.cross(C.k) == -cos(q3) * C.j
    assert C.i.cross(A.i) == -sin(q3) * cos(q2) * A.j + -sin(q2) * sin(q3) * A.k
    assert C.j.cross(A.i) == sin(q2) * A.j + -cos(q2) * A.k
    assert express(C.k.cross(A.i), C).trigsimp() == cos(q3) * C.j