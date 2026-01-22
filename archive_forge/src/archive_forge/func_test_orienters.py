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
def test_orienters():
    A = CoordSys3D('A')
    axis_orienter = AxisOrienter(a, A.k)
    body_orienter = BodyOrienter(a, b, c, '123')
    space_orienter = SpaceOrienter(a, b, c, '123')
    q_orienter = QuaternionOrienter(q1, q2, q3, q4)
    assert axis_orienter.rotation_matrix(A) == Matrix([[cos(a), sin(a), 0], [-sin(a), cos(a), 0], [0, 0, 1]])
    assert body_orienter.rotation_matrix() == Matrix([[cos(b) * cos(c), sin(a) * sin(b) * cos(c) + sin(c) * cos(a), sin(a) * sin(c) - sin(b) * cos(a) * cos(c)], [-sin(c) * cos(b), -sin(a) * sin(b) * sin(c) + cos(a) * cos(c), sin(a) * cos(c) + sin(b) * sin(c) * cos(a)], [sin(b), -sin(a) * cos(b), cos(a) * cos(b)]])
    assert space_orienter.rotation_matrix() == Matrix([[cos(b) * cos(c), sin(c) * cos(b), -sin(b)], [sin(a) * sin(b) * cos(c) - sin(c) * cos(a), sin(a) * sin(b) * sin(c) + cos(a) * cos(c), sin(a) * cos(b)], [sin(a) * sin(c) + sin(b) * cos(a) * cos(c), -sin(a) * cos(c) + sin(b) * sin(c) * cos(a), cos(a) * cos(b)]])
    assert q_orienter.rotation_matrix() == Matrix([[q1 ** 2 + q2 ** 2 - q3 ** 2 - q4 ** 2, 2 * q1 * q4 + 2 * q2 * q3, -2 * q1 * q3 + 2 * q2 * q4], [-2 * q1 * q4 + 2 * q2 * q3, q1 ** 2 - q2 ** 2 + q3 ** 2 - q4 ** 2, 2 * q1 * q2 + 2 * q3 * q4], [2 * q1 * q3 + 2 * q2 * q4, -2 * q1 * q2 + 2 * q3 * q4, q1 ** 2 - q2 ** 2 - q3 ** 2 + q4 ** 2]])