from sympy.physics.vector import dynamicsymbols, Point, ReferenceFrame
from sympy.testing.pytest import raises, ignore_warnings
import warnings
def test_point_v1pt_theorys():
    q, q2 = dynamicsymbols('q q2')
    qd, q2d = dynamicsymbols('q q2', 1)
    qdd, q2dd = dynamicsymbols('q q2', 2)
    N = ReferenceFrame('N')
    B = ReferenceFrame('B')
    B.set_ang_vel(N, qd * B.z)
    O = Point('O')
    P = O.locatenew('P', B.x)
    P.set_vel(B, 0)
    O.set_vel(N, 0)
    assert P.v1pt_theory(O, N, B) == qd * B.y
    O.set_vel(N, N.x)
    assert P.v1pt_theory(O, N, B) == N.x + qd * B.y
    P.set_vel(B, B.z)
    assert P.v1pt_theory(O, N, B) == B.z + N.x + qd * B.y