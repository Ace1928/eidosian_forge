from sympy.tensor.functions import TensorProduct
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.tensor.array import Array
from sympy.abc import x, y, z
from sympy.abc import i, j, k, l
def test_TensorProduct_construction():
    assert TensorProduct(3, 4) == 12
    assert isinstance(TensorProduct(A, A), TensorProduct)
    expr = TensorProduct(TensorProduct(x, y), z)
    assert expr == x * y * z
    expr = TensorProduct(TensorProduct(A, B), C)
    assert expr == TensorProduct(A, B, C)
    expr = TensorProduct(Matrix.eye(2), Array([[0, -1], [1, 0]]))
    assert expr == Array([[[[0, -1], [1, 0]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[0, -1], [1, 0]]]])