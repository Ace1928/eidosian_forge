from sympy.core.backend import sin, cos, tan, pi, symbols, Matrix, S, Function
from sympy.physics.mechanics import (Particle, Point, ReferenceFrame,
from sympy.physics.mechanics import (angular_momentum, dynamicsymbols,
from sympy.physics.mechanics.functions import (gravity, center_of_mass,
from sympy.testing.pytest import raises
def test_inertia_of_point_mass():
    r, s, t, m = symbols('r s t m')
    N = ReferenceFrame('N')
    px = r * N.x
    I = inertia_of_point_mass(m, px, N)
    assert I == m * r ** 2 * (N.y | N.y) + m * r ** 2 * (N.z | N.z)
    py = s * N.y
    I = inertia_of_point_mass(m, py, N)
    assert I == m * s ** 2 * (N.x | N.x) + m * s ** 2 * (N.z | N.z)
    pz = t * N.z
    I = inertia_of_point_mass(m, pz, N)
    assert I == m * t ** 2 * (N.x | N.x) + m * t ** 2 * (N.y | N.y)
    p = px + py + pz
    I = inertia_of_point_mass(m, p, N)
    assert I == m * (s ** 2 + t ** 2) * (N.x | N.x) - m * r * s * (N.x | N.y) - m * r * t * (N.x | N.z) - m * r * s * (N.y | N.x) + m * (r ** 2 + t ** 2) * (N.y | N.y) - m * s * t * (N.y | N.z) - m * r * t * (N.z | N.x) - m * s * t * (N.z | N.y) + m * (r ** 2 + s ** 2) * (N.z | N.z)