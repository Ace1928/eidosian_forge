from sympy.physics.vector import dynamicsymbols, Point, ReferenceFrame
from sympy.testing.pytest import raises, ignore_warnings
import warnings
def test_point_partial_velocity():
    N = ReferenceFrame('N')
    A = ReferenceFrame('A')
    p = Point('p')
    u1, u2 = dynamicsymbols('u1, u2')
    p.set_vel(N, u1 * A.x + u2 * N.y)
    assert p.partial_velocity(N, u1) == A.x
    assert p.partial_velocity(N, u1, u2) == (A.x, N.y)
    raises(ValueError, lambda: p.partial_velocity(A, u1))