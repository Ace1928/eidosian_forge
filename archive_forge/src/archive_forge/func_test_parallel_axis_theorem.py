from sympy.physics.matrices import msigma, mgamma, minkowski_tensor, pat_matrix, mdft
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import (Matrix, eye, zeros)
from sympy.testing.pytest import warns_deprecated_sympy
def test_parallel_axis_theorem():
    mat1 = Matrix(((2, -1, -1), (-1, 2, -1), (-1, -1, 2)))
    assert pat_matrix(1, 1, 1, 1) == mat1
    assert pat_matrix(2, 1, 1, 1) == 2 * mat1
    mat2 = Matrix(((0, 0, 0), (0, 1, 0), (0, 0, 1)))
    assert pat_matrix(1, 1, 0, 0) == mat2
    assert pat_matrix(2, 1, 0, 0) == 2 * mat2
    mat3 = Matrix(((1, 0, 0), (0, 0, 0), (0, 0, 1)))
    assert pat_matrix(1, 0, 1, 0) == mat3
    assert pat_matrix(2, 0, 1, 0) == 2 * mat3
    mat4 = Matrix(((1, 0, 0), (0, 1, 0), (0, 0, 0)))
    assert pat_matrix(1, 0, 0, 1) == mat4
    assert pat_matrix(2, 0, 0, 1) == 2 * mat4