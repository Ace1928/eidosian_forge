from sympy.physics.vector import dynamicsymbols, Point, ReferenceFrame
from sympy.testing.pytest import raises, ignore_warnings
import warnings
def test_auto_vel_derivative():
    q1, q2 = dynamicsymbols('q1:3')
    u1, u2 = dynamicsymbols('u1:3', 1)
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    C = ReferenceFrame('C')
    B.orient_axis(A, A.z, q1)
    B.set_ang_vel(A, u1 * A.z)
    C.orient_axis(B, B.z, q2)
    C.set_ang_vel(B, u2 * B.z)
    Am = Point('Am')
    Am.set_vel(A, 0)
    Bm = Point('Bm')
    Bm.set_pos(Am, B.x)
    Bm.set_vel(B, 0)
    Bm.set_vel(C, 0)
    Cm = Point('Cm')
    Cm.set_pos(Bm, C.x)
    Cm.set_vel(C, 0)
    temp = Cm._vel_dict.copy()
    assert Cm.vel(A) == u1 * B.y + (u1 + u2) * C.y
    Cm._vel_dict = temp
    Cm.v2pt_theory(Bm, B, C)
    assert Cm.vel(A) == u1 * B.y + (u1 + u2) * C.y