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
def test_parametric_surfaceintegrals():
    semisphere = ParametricRegion((2 * sin(phi) * cos(theta), 2 * sin(phi) * sin(theta), 2 * cos(phi)), (theta, 0, 2 * pi), (phi, 0, pi / 2))
    assert ParametricIntegral(C.z, semisphere) == 8 * pi
    cylinder = ParametricRegion((sqrt(3) * cos(theta), sqrt(3) * sin(theta), z), (z, 0, 6), (theta, 0, 2 * pi))
    assert ParametricIntegral(C.y, cylinder) == 0
    cone = ParametricRegion((v * cos(u), v * sin(u), v), (u, 0, 2 * pi), (v, 0, 1))
    assert ParametricIntegral(C.x * C.i + C.y * C.j + C.z ** 4 * C.k, cone) == pi / 3
    triangle1 = ParametricRegion((x, y), (x, 0, 2), (y, 0, 10 - 5 * x))
    triangle2 = ParametricRegion((x, y), (y, 0, 10 - 5 * x), (x, 0, 2))
    assert ParametricIntegral(-15.6 * C.y * C.k, triangle1) == ParametricIntegral(-15.6 * C.y * C.k, triangle2)
    assert ParametricIntegral(C.z, triangle1) == 10 * C.z