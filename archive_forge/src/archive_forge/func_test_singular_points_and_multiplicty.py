from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.abc import x, y, z, s, t
from sympy.sets import FiniteSet, EmptySet
from sympy.geometry import Point
from sympy.vector import ImplicitRegion
from sympy.testing.pytest import raises
def test_singular_points_and_multiplicty():
    r1 = ImplicitRegion((x, y, z), Eq(x + y + z, 0))
    assert r1.singular_points() == EmptySet
    r2 = ImplicitRegion((x, y, z), x * y * z + y ** 4 - x ** 2 * z ** 2)
    assert r2.singular_points() == FiniteSet((0, 0, z), (x, 0, 0))
    assert r2.multiplicity((0, 0, 0)) == 3
    assert r2.multiplicity((0, 0, 6)) == 2
    r3 = ImplicitRegion((x, y, z), z ** 2 - x ** 2 - y ** 2)
    assert r3.singular_points() == FiniteSet((0, 0, 0))
    assert r3.multiplicity((0, 0, 0)) == 2
    r4 = ImplicitRegion((x, y), x ** 2 + y ** 2 - 2 * x)
    assert r4.singular_points() == EmptySet
    assert r4.multiplicity(Point(1, 3)) == 0