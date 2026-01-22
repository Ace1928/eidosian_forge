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
def test_tensorflow_printing():
    assert tensorflow_code(eye(3)) == 'tensorflow.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])'
    expr = Matrix([[x, sin(y)], [exp(z), -t]])
    assert tensorflow_code(expr) == 'tensorflow.Variable([[x, tensorflow.math.sin(y)], [tensorflow.math.exp(z), -t]])'