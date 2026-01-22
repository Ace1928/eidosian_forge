from sympy.physics.vector import dynamicsymbols, Point, ReferenceFrame
from sympy.testing.pytest import raises, ignore_warnings
import warnings
def test_auto_point_acc_zero_vel():
    N = ReferenceFrame('N')
    O = Point('O')
    O.set_vel(N, 0)
    assert O.acc(N) == 0 * N.x