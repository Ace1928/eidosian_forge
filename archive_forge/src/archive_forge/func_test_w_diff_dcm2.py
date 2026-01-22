from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import (eye, zeros)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.simplify.simplify import simplify
from sympy.physics.vector import (ReferenceFrame, Vector, CoordinateSym,
from sympy.physics.vector.frame import _check_frame
from sympy.physics.vector.vector import VectorTypeError
from sympy.testing.pytest import raises
import warnings
def test_w_diff_dcm2():
    q1, q2, q3 = dynamicsymbols('q1:4')
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'axis', [q1, N.x])
    B = A.orientnew('B', 'axis', [q2, A.y])
    C = B.orientnew('C', 'axis', [q3, B.z])
    DCM = C.dcm(N).T
    D = N.orientnew('D', 'DCM', DCM)
    assert D.dcm(N) == C.dcm(N) == Matrix([[cos(q2) * cos(q3), sin(q1) * sin(q2) * cos(q3) + sin(q3) * cos(q1), sin(q1) * sin(q3) - sin(q2) * cos(q1) * cos(q3)], [-sin(q3) * cos(q2), -sin(q1) * sin(q2) * sin(q3) + cos(q1) * cos(q3), sin(q1) * cos(q3) + sin(q2) * sin(q3) * cos(q1)], [sin(q2), -sin(q1) * cos(q2), cos(q1) * cos(q2)]])
    assert (D.ang_vel_in(N) - C.ang_vel_in(N)).express(N).simplify() == 0