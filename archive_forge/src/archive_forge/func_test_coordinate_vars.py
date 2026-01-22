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
def test_coordinate_vars():
    """
    Tests the coordinate variables functionality with respect to
    reorientation of coordinate systems.
    """
    A = CoordSys3D('A')
    assert BaseScalar(0, A, 'A_x', '\\mathbf{{x}_{A}}') == A.x
    assert BaseScalar(1, A, 'A_y', '\\mathbf{{y}_{A}}') == A.y
    assert BaseScalar(2, A, 'A_z', '\\mathbf{{z}_{A}}') == A.z
    assert BaseScalar(0, A, 'A_x', '\\mathbf{{x}_{A}}').__hash__() == A.x.__hash__()
    assert isinstance(A.x, BaseScalar) and isinstance(A.y, BaseScalar) and isinstance(A.z, BaseScalar)
    assert A.x * A.y == A.y * A.x
    assert A.scalar_map(A) == {A.x: A.x, A.y: A.y, A.z: A.z}
    assert A.x.system == A
    assert A.x.diff(A.x) == 1
    B = A.orient_new_axis('B', q, A.k)
    assert B.scalar_map(A) == {B.z: A.z, B.y: -A.x * sin(q) + A.y * cos(q), B.x: A.x * cos(q) + A.y * sin(q)}
    assert A.scalar_map(B) == {A.x: B.x * cos(q) - B.y * sin(q), A.y: B.x * sin(q) + B.y * cos(q), A.z: B.z}
    assert express(B.x, A, variables=True) == A.x * cos(q) + A.y * sin(q)
    assert express(B.y, A, variables=True) == -A.x * sin(q) + A.y * cos(q)
    assert express(B.z, A, variables=True) == A.z
    assert expand(express(B.x * B.y * B.z, A, variables=True)) == expand(A.z * (-A.x * sin(q) + A.y * cos(q)) * (A.x * cos(q) + A.y * sin(q)))
    assert express(B.x * B.i + B.y * B.j + B.z * B.k, A) == (B.x * cos(q) - B.y * sin(q)) * A.i + (B.x * sin(q) + B.y * cos(q)) * A.j + B.z * A.k
    assert simplify(express(B.x * B.i + B.y * B.j + B.z * B.k, A, variables=True)) == A.x * A.i + A.y * A.j + A.z * A.k
    assert express(A.x * A.i + A.y * A.j + A.z * A.k, B) == (A.x * cos(q) + A.y * sin(q)) * B.i + (-A.x * sin(q) + A.y * cos(q)) * B.j + A.z * B.k
    assert simplify(express(A.x * A.i + A.y * A.j + A.z * A.k, B, variables=True)) == B.x * B.i + B.y * B.j + B.z * B.k
    N = B.orient_new_axis('N', -q, B.k)
    assert N.scalar_map(A) == {N.x: A.x, N.z: A.z, N.y: A.y}
    C = A.orient_new_axis('C', q, A.i + A.j + A.k)
    mapping = A.scalar_map(C)
    assert mapping[A.x].equals(C.x * (2 * cos(q) + 1) / 3 + C.y * (-2 * sin(q + pi / 6) + 1) / 3 + C.z * (-2 * cos(q + pi / 3) + 1) / 3)
    assert mapping[A.y].equals(C.x * (-2 * cos(q + pi / 3) + 1) / 3 + C.y * (2 * cos(q) + 1) / 3 + C.z * (-2 * sin(q + pi / 6) + 1) / 3)
    assert mapping[A.z].equals(C.x * (-2 * sin(q + pi / 6) + 1) / 3 + C.y * (-2 * cos(q + pi / 3) + 1) / 3 + C.z * (2 * cos(q) + 1) / 3)
    D = A.locate_new('D', a * A.i + b * A.j + c * A.k)
    assert D.scalar_map(A) == {D.z: A.z - c, D.x: A.x - a, D.y: A.y - b}
    E = A.orient_new_axis('E', a, A.k, a * A.i + b * A.j + c * A.k)
    assert A.scalar_map(E) == {A.z: E.z + c, A.x: E.x * cos(a) - E.y * sin(a) + a, A.y: E.x * sin(a) + E.y * cos(a) + b}
    assert E.scalar_map(A) == {E.x: (A.x - a) * cos(a) + (A.y - b) * sin(a), E.y: (-A.x + a) * sin(a) + (A.y - b) * cos(a), E.z: A.z - c}
    F = A.locate_new('F', Vector.zero)
    assert A.scalar_map(F) == {A.z: F.z, A.x: F.x, A.y: F.y}