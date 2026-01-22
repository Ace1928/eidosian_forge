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
def test_orientnew_respects_input_variables():
    N = ReferenceFrame('N')
    q1 = dynamicsymbols('q1')
    A = N.orientnew('a', 'Axis', [q1, N.z])
    name = 'b'
    new_variables = ['notb_' + x + '1' for x in N.indices]
    B = N.orientnew(name, 'Axis', [q1, N.z], variables=new_variables)
    for j, var in enumerate(A.varlist):
        assert var.name == A.name + '_' + A.indices[j]
    for j, var in enumerate(B.varlist):
        assert var.name == new_variables[j]