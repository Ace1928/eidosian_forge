from sympy.core.numbers import (Float, pi)
from sympy.core.symbol import symbols
from sympy.core.sorting import ordered
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.physics.vector import ReferenceFrame, Vector, dynamicsymbols, dot
from sympy.physics.vector.vector import VectorTypeError
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
def test_Vector():
    assert A.x != A.y
    assert A.y != A.z
    assert A.z != A.x
    assert A.x + 0 == A.x
    v1 = x * A.x + y * A.y + z * A.z
    v2 = x ** 2 * A.x + y ** 2 * A.y + z ** 2 * A.z
    v3 = v1 + v2
    v4 = v1 - v2
    assert isinstance(v1, Vector)
    assert dot(v1, A.x) == x
    assert dot(v1, A.y) == y
    assert dot(v1, A.z) == z
    assert isinstance(v2, Vector)
    assert dot(v2, A.x) == x ** 2
    assert dot(v2, A.y) == y ** 2
    assert dot(v2, A.z) == z ** 2
    assert isinstance(v3, Vector)
    assert dot(v3, A.x) == x ** 2 + x
    assert dot(v3, A.y) == y ** 2 + y
    assert dot(v3, A.z) == z ** 2 + z
    assert isinstance(v4, Vector)
    assert dot(v4, A.x) == x - x ** 2
    assert dot(v4, A.y) == y - y ** 2
    assert dot(v4, A.z) == z - z ** 2
    assert v1.to_matrix(A) == Matrix([[x], [y], [z]])
    q = symbols('q')
    B = A.orientnew('B', 'Axis', (q, A.x))
    assert v1.to_matrix(B) == Matrix([[x], [y * cos(q) + z * sin(q)], [-y * sin(q) + z * cos(q)]])
    B = ReferenceFrame('B')
    v5 = x * A.x + y * A.y + z * B.z
    assert Vector(0).separate() == {}
    assert v1.separate() == {A: v1}
    assert v5.separate() == {A: x * A.x + y * A.y, B: z * B.z}
    v6 = x * A.x + y * A.y + z * A.z
    assert v6.free_symbols(A) == {x, y, z}
    raises(TypeError, lambda: v3.applyfunc(v1))