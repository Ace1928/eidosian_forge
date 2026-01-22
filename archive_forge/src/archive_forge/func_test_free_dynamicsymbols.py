from sympy.core.numbers import (Float, pi)
from sympy.core.symbol import symbols
from sympy.core.sorting import ordered
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.physics.vector import ReferenceFrame, Vector, dynamicsymbols, dot
from sympy.physics.vector.vector import VectorTypeError
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
def test_free_dynamicsymbols():
    A, B, C, D = symbols('A, B, C, D', cls=ReferenceFrame)
    a, b, c, d, e, f = dynamicsymbols('a, b, c, d, e, f')
    B.orient_axis(A, a, A.x)
    C.orient_axis(B, b, B.y)
    D.orient_axis(C, c, C.x)
    v = d * D.x + e * D.y + f * D.z
    assert set(ordered(v.free_dynamicsymbols(A))) == {a, b, c, d, e, f}
    assert set(ordered(v.free_dynamicsymbols(B))) == {b, c, d, e, f}
    assert set(ordered(v.free_dynamicsymbols(C))) == {c, d, e, f}
    assert set(ordered(v.free_dynamicsymbols(D))) == {d, e, f}