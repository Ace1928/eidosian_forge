from sympy.core.function import expand_mul
from sympy.core.numbers import I, Rational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.complexes import Abs
from sympy.simplify.simplify import simplify
from sympy.matrices.matrices import NonSquareMatrixError
from sympy.matrices import Matrix, zeros, eye, SparseMatrix
from sympy.abc import x, y, z
from sympy.testing.pytest import raises, slow
from sympy.testing.matrices import allclose
def test_singular_value_decompositionD():
    A = Matrix([[1, 2], [2, 1]])
    U, S, V = A.singular_value_decomposition()
    assert U * S * V.T == A
    assert U.T * U == eye(U.cols)
    assert V.T * V == eye(V.cols)
    B = Matrix([[1, 2]])
    U, S, V = B.singular_value_decomposition()
    assert U * S * V.T == B
    assert U.T * U == eye(U.cols)
    assert V.T * V == eye(V.cols)
    C = Matrix([[1, 0, 0, 0, 2], [0, 0, 3, 0, 0], [0, 0, 0, 0, 0], [0, 2, 0, 0, 0]])
    U, S, V = C.singular_value_decomposition()
    assert U * S * V.T == C
    assert U.T * U == eye(U.cols)
    assert V.T * V == eye(V.cols)
    D = Matrix([[Rational(1, 3), sqrt(2)], [0, Rational(1, 4)]])
    U, S, V = D.singular_value_decomposition()
    assert simplify(U.T * U) == eye(U.cols)
    assert simplify(V.T * V) == eye(V.cols)
    assert simplify(U * S * V.T) == D