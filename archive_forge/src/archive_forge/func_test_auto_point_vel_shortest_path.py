from sympy.physics.vector import dynamicsymbols, Point, ReferenceFrame
from sympy.testing.pytest import raises, ignore_warnings
import warnings
def test_auto_point_vel_shortest_path():
    t = dynamicsymbols._t
    q1, q2, u1, u2 = dynamicsymbols('q1 q2 u1 u2')
    B = ReferenceFrame('B')
    P = Point('P')
    P.set_vel(B, u1 * B.x)
    P1 = Point('P1')
    P1.set_pos(P, q2 * B.y)
    P1.set_vel(B, q1 * B.z)
    P2 = Point('P2')
    P2.set_pos(P1, q1 * B.z)
    P3 = Point('P3')
    P3.set_pos(P2, 10 * q1 * B.y)
    P4 = Point('P4')
    P4.set_pos(P3, q1 * B.x)
    O = Point('O')
    O.set_vel(B, u2 * B.y)
    O1 = Point('O1')
    O1.set_pos(O, q2 * B.z)
    P4.set_pos(O1, q1 * B.x + q2 * B.z)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with ignore_warnings(UserWarning):
            assert P4.vel(B) == q1.diff(t) * B.x + u2 * B.y + 2 * q2.diff(t) * B.z