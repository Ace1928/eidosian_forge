from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.testing.pytest import raises
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.integrals import ParametricIntegral, vector_integrate
from sympy.vector.parametricregion import ParametricRegion
from sympy.vector.implicitregion import ImplicitRegion
from sympy.abc import x, y, z, u, v, r, t, theta, phi
from sympy.geometry import Point, Segment, Curve, Circle, Polygon, Plane
def test_parametric_lineintegrals():
    halfcircle = ParametricRegion((4 * cos(theta), 4 * sin(theta)), (theta, -pi / 2, pi / 2))
    assert ParametricIntegral(C.x * C.y ** 4, halfcircle) == S(8192) / 5
    curve = ParametricRegion((t, t ** 2, t ** 3), (t, 0, 1))
    field1 = 8 * C.x ** 2 * C.y * C.z * C.i + 5 * C.z * C.j - 4 * C.x * C.y * C.k
    assert ParametricIntegral(field1, curve) == 1
    line = ParametricRegion((4 * t - 1, 2 - 2 * t, t), (t, 0, 1))
    assert ParametricIntegral(C.x * C.z * C.i - C.y * C.z * C.k, line) == 3
    assert ParametricIntegral(4 * C.x ** 3, ParametricRegion((1, t), (t, 0, 2))) == 8
    helix = ParametricRegion((cos(t), sin(t), 3 * t), (t, 0, 4 * pi))
    assert ParametricIntegral(C.x * C.y * C.z, helix) == -3 * sqrt(10) * pi
    field2 = C.y * C.i + C.z * C.j + C.z * C.k
    assert ParametricIntegral(field2, ParametricRegion((cos(t), sin(t), t ** 2), (t, 0, pi))) == -5 * pi / 2 + pi ** 4 / 2