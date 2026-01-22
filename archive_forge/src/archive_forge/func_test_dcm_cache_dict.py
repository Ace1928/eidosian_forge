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
def test_dcm_cache_dict():
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    C = ReferenceFrame('C')
    D = ReferenceFrame('D')
    a, b, c = symbols('a b c')
    B.orient_axis(A, A.x, a)
    C.orient_axis(B, B.x, b)
    D.orient_axis(C, C.x, c)
    assert D._dcm_dict == {C: Matrix([[1, 0, 0], [0, cos(c), sin(c)], [0, -sin(c), cos(c)]])}
    assert C._dcm_dict == {B: Matrix([[1, 0, 0], [0, cos(b), sin(b)], [0, -sin(b), cos(b)]]), D: Matrix([[1, 0, 0], [0, cos(c), -sin(c)], [0, sin(c), cos(c)]])}
    assert B._dcm_dict == {A: Matrix([[1, 0, 0], [0, cos(a), sin(a)], [0, -sin(a), cos(a)]]), C: Matrix([[1, 0, 0], [0, cos(b), -sin(b)], [0, sin(b), cos(b)]])}
    assert A._dcm_dict == {B: Matrix([[1, 0, 0], [0, cos(a), -sin(a)], [0, sin(a), cos(a)]])}
    assert D._dcm_dict == D._dcm_cache
    D.dcm(A)
    assert list(A._dcm_cache.keys()) == [A, B, D]
    assert list(D._dcm_cache.keys()) == [C, A]
    assert list(A._dcm_dict.keys()) == [B]
    assert list(D._dcm_dict.keys()) == [C]
    assert A._dcm_dict != A._dcm_cache
    A.orient_axis(B, B.x, b)
    assert A._dcm_dict == {B: Matrix([[1, 0, 0], [0, cos(b), sin(b)], [0, -sin(b), cos(b)]])}
    assert A._dcm_dict == A._dcm_cache
    assert B._dcm_dict == {C: Matrix([[1, 0, 0], [0, cos(b), -sin(b)], [0, sin(b), cos(b)]]), A: Matrix([[1, 0, 0], [0, cos(b), -sin(b)], [0, sin(b), cos(b)]])}