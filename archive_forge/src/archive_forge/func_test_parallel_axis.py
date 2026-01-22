from sympy.core.symbol import symbols
from sympy.physics.mechanics import Point, ReferenceFrame, Dyadic, RigidBody
from sympy.physics.mechanics import dynamicsymbols, outer, inertia
from sympy.physics.mechanics import inertia_of_point_mass
from sympy.core.backend import expand, zeros, _simplify_matrix
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_parallel_axis():
    N = ReferenceFrame('N')
    m, Ix, Iy, Iz, a, b = symbols('m, I_x, I_y, I_z, a, b')
    Io = inertia(N, Ix, Iy, Iz)
    o = Point('o')
    p = o.locatenew('p', a * N.x + b * N.y)
    R = RigidBody('R', o, N, m, (Io, o))
    Ip = R.parallel_axis(p)
    Ip_expected = inertia(N, Ix + m * b ** 2, Iy + m * a ** 2, Iz + m * (a ** 2 + b ** 2), ixy=-m * a * b)
    assert Ip == Ip_expected
    A = ReferenceFrame('A')
    A.orient_axis(N, N.z, 1)
    assert _simplify_matrix((R.parallel_axis(p, A) - Ip_expected).to_matrix(A)) == zeros(3, 3)