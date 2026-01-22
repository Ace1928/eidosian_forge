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
def test_dcm_diff_16824():
    q1, q2, q3 = dynamicsymbols('q1:4')
    s1 = sin(q1)
    c1 = cos(q1)
    s2 = sin(q2)
    c2 = cos(q2)
    s3 = sin(q3)
    c3 = cos(q3)
    dcm = Matrix([[c2 * c3, s1 * s2 * c3 - s3 * c1, c1 * s2 * c3 + s3 * s1], [c2 * s3, s1 * s2 * s3 + c3 * c1, c1 * s2 * s3 - c3 * s1], [-s2, s1 * c2, c1 * c2]])
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    B.orient(A, 'DCM', dcm)
    AwB = B.ang_vel_in(A)
    alpha2 = s3 * c2 * q1.diff() + c3 * q2.diff()
    beta2 = s1 * c2 * q3.diff() + c1 * q2.diff()
    assert simplify(AwB.dot(A.y) - alpha2) == 0
    assert simplify(AwB.dot(B.y) - beta2) == 0