from sympy.core.numbers import (Float, pi)
from sympy.core.symbol import symbols
from sympy.core.sorting import ordered
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.physics.vector import ReferenceFrame, Vector, dynamicsymbols, dot
from sympy.physics.vector.vector import VectorTypeError
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
def test_vector_angle():
    A = ReferenceFrame('A')
    v1 = A.x + A.y
    v2 = A.z
    assert v1.angle_between(v2) == pi / 2
    B = ReferenceFrame('B')
    B.orient_axis(A, A.x, pi)
    v3 = A.x
    v4 = B.x
    assert v3.angle_between(v4) == 0