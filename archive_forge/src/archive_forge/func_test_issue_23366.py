from sympy.core.numbers import (Float, pi)
from sympy.core.symbol import symbols
from sympy.core.sorting import ordered
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.physics.vector import ReferenceFrame, Vector, dynamicsymbols, dot
from sympy.physics.vector.vector import VectorTypeError
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
def test_issue_23366():
    u1 = dynamicsymbols('u1')
    N = ReferenceFrame('N')
    N_v_A = u1 * N.x
    raises(VectorTypeError, lambda: N_v_A.diff(N, u1))