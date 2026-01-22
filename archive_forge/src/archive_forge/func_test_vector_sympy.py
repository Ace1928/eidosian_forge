from sympy.core import Rational, S
from sympy.simplify import simplify, trigsimp
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.vector.vector import Vector, BaseVector, VectorAdd, \
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.vector import Cross, Dot, cross
from sympy.testing.pytest import raises
def test_vector_sympy():
    """
    Test whether the Vector framework confirms to the hashing
    and equality testing properties of SymPy.
    """
    v1 = 3 * j
    assert v1 == j * 3
    assert v1.components == {j: 3}
    v2 = 3 * i + 4 * j + 5 * k
    v3 = 2 * i + 4 * j + i + 4 * k + k
    assert v3 == v2
    assert v3.__hash__() == v2.__hash__()