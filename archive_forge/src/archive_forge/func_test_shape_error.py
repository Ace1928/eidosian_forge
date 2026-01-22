from sympy.matrices.dense import Matrix, eye
from sympy.matrices.common import ShapeError
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.special import Identity, OneMatrix, ZeroMatrix
from sympy.core import symbols
from sympy.testing.pytest import raises, warns_deprecated_sympy
from sympy.matrices import MatrixSymbol
from sympy.matrices.expressions import (HadamardProduct, hadamard_product, HadamardPower, hadamard_power)
def test_shape_error():
    A = MatrixSymbol('A', 2, 3)
    B = MatrixSymbol('B', 3, 3)
    raises(ShapeError, lambda: HadamardProduct(A, B))
    raises(ShapeError, lambda: HadamardPower(A, B))
    A = MatrixSymbol('A', 3, 2)
    raises(ShapeError, lambda: HadamardProduct(A, B))
    raises(ShapeError, lambda: HadamardPower(A, B))