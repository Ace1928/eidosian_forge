from sympy.core.numbers import comp, Rational
from sympy.physics.optics.utils import (refraction_angle, fresnel_coefficients,
from sympy.physics.optics.medium import Medium
from sympy.physics.units import e0
from sympy.core.numbers import oo
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.geometry.point import Point3D
from sympy.geometry.line import Ray3D
from sympy.geometry.plane import Plane
from sympy.testing.pytest import raises
def test_refraction_angle():
    n1, n2 = symbols('n1, n2')
    m1 = Medium('m1')
    m2 = Medium('m2')
    r1 = Ray3D(Point3D(-1, -1, 1), Point3D(0, 0, 0))
    i = Matrix([1, 1, 1])
    n = Matrix([0, 0, 1])
    normal_ray = Ray3D(Point3D(0, 0, 0), Point3D(0, 0, 1))
    P = Plane(Point3D(0, 0, 0), normal_vector=[0, 0, 1])
    assert refraction_angle(r1, 1, 1, n) == Matrix([[1], [1], [-1]])
    assert refraction_angle([1, 1, 1], 1, 1, n) == Matrix([[1], [1], [-1]])
    assert refraction_angle((1, 1, 1), 1, 1, n) == Matrix([[1], [1], [-1]])
    assert refraction_angle(i, 1, 1, [0, 0, 1]) == Matrix([[1], [1], [-1]])
    assert refraction_angle(i, 1, 1, (0, 0, 1)) == Matrix([[1], [1], [-1]])
    assert refraction_angle(i, 1, 1, normal_ray) == Matrix([[1], [1], [-1]])
    assert refraction_angle(i, 1, 1, plane=P) == Matrix([[1], [1], [-1]])
    assert refraction_angle(r1, 1, 1, plane=P) == Ray3D(Point3D(0, 0, 0), Point3D(1, 1, -1))
    assert refraction_angle(r1, m1, 1.33, plane=P) == Ray3D(Point3D(0, 0, 0), Point3D(Rational(100, 133), Rational(100, 133), -789378201649271 * sqrt(3) / 1000000000000000))
    assert refraction_angle(r1, 1, m2, plane=P) == Ray3D(Point3D(0, 0, 0), Point3D(1, 1, -1))
    assert refraction_angle(r1, n1, n2, plane=P) == Ray3D(Point3D(0, 0, 0), Point3D(n1 / n2, n1 / n2, -sqrt(3) * sqrt(-2 * n1 ** 2 / (3 * n2 ** 2) + 1)))
    assert refraction_angle(r1, 1.33, 1, plane=P) == 0
    assert refraction_angle(r1, 1, 1, normal_ray) == Ray3D(Point3D(0, 0, 0), direction_ratio=[1, 1, -1])
    assert ae(refraction_angle(0.5, 1, 2), 0.24207, 5)
    assert ae(refraction_angle(0.5, 2, 1), 1.28293, 5)
    raises(ValueError, lambda: refraction_angle(r1, m1, m2, normal_ray, P))
    raises(TypeError, lambda: refraction_angle(m1, m1, m2))
    raises(TypeError, lambda: refraction_angle(r1, m1, m2, None, i))
    raises(TypeError, lambda: refraction_angle(r1, m1, m2, m2))