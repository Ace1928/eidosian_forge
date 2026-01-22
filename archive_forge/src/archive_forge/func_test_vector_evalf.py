from sympy.core.numbers import (Float, pi)
from sympy.core.symbol import symbols
from sympy.core.sorting import ordered
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.physics.vector import ReferenceFrame, Vector, dynamicsymbols, dot
from sympy.physics.vector.vector import VectorTypeError
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
def test_vector_evalf():
    a, b = symbols('a b')
    v = pi * A.x
    assert v.evalf(2) == Float('3.1416', 2) * A.x
    v = pi * A.x + 5 * a * A.y - b * A.z
    assert v.evalf(3) == Float('3.1416', 3) * A.x + Float('5', 3) * a * A.y - b * A.z
    assert v.evalf(5, subs={a: 1.234, b: 5.8973}) == Float('3.1415926536', 5) * A.x + Float('6.17', 5) * A.y - Float('5.8973', 5) * A.z