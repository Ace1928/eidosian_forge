import random
from sympy.core.function import Derivative
from sympy.core.symbol import symbols
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, ArrayAdd, \
from sympy.core.relational import Eq, Ne, Ge, Gt, Le, Lt
from sympy.external import import_module
from sympy.functions import \
from sympy.matrices import Matrix, MatrixBase, eye, randMatrix
from sympy.matrices.expressions import \
from sympy.printing.tensorflow import tensorflow_code
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
from sympy.utilities.lambdify import lambdify
from sympy.testing.pytest import skip
from sympy.testing.pytest import XFAIL
@XFAIL
def test_tensorflow_matrices():
    if not tf:
        skip('TensorFlow not installed')
    expr = M
    assert tensorflow_code(expr) == 'M'
    _compare_tensorflow_matrix((M,), expr)
    expr = M + N
    assert tensorflow_code(expr) == 'tensorflow.math.add(M, N)'
    _compare_tensorflow_matrix((M, N), expr)
    expr = M * N
    assert tensorflow_code(expr) == 'tensorflow.linalg.matmul(M, N)'
    _compare_tensorflow_matrix((M, N), expr)
    expr = HadamardProduct(M, N)
    assert tensorflow_code(expr) == 'tensorflow.math.multiply(M, N)'
    _compare_tensorflow_matrix((M, N), expr)
    expr = M * N * P * Q
    assert tensorflow_code(expr) == 'tensorflow.linalg.matmul(tensorflow.linalg.matmul(tensorflow.linalg.matmul(M, N), P), Q)'
    _compare_tensorflow_matrix((M, N, P, Q), expr)
    expr = M ** 3
    assert tensorflow_code(expr) == 'tensorflow.linalg.matmul(tensorflow.linalg.matmul(M, M), M)'
    _compare_tensorflow_matrix((M,), expr)
    expr = Trace(M)
    assert tensorflow_code(expr) == 'tensorflow.linalg.trace(M)'
    _compare_tensorflow_matrix((M,), expr)
    expr = Determinant(M)
    assert tensorflow_code(expr) == 'tensorflow.linalg.det(M)'
    _compare_tensorflow_matrix_scalar((M,), expr)
    expr = Inverse(M)
    assert tensorflow_code(expr) == 'tensorflow.linalg.inv(M)'
    _compare_tensorflow_matrix_inverse((M,), expr, use_float=True)
    expr = M.T
    assert tensorflow_code(expr, tensorflow_version='1.14') == 'tensorflow.linalg.matrix_transpose(M)'
    assert tensorflow_code(expr, tensorflow_version='1.13') == 'tensorflow.matrix_transpose(M)'
    _compare_tensorflow_matrix((M,), expr)