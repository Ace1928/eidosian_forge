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
def test_orient_axis():
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    A.orient_axis(B, -B.x, 1)
    A1 = A.dcm(B)
    A.orient_axis(B, B.x, -1)
    A2 = A.dcm(B)
    A.orient_axis(B, 1, -B.x)
    A3 = A.dcm(B)
    assert A1 == A2
    assert A2 == A3
    raises(TypeError, lambda: A.orient_axis(B, 1, 1))