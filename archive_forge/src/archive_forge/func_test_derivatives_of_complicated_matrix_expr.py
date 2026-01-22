from sympy import KroneckerProduct
from sympy.combinatorics import Permutation
from sympy.concrete.summations import Sum
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.expressions.determinant import Determinant
from sympy.matrices.expressions.diagonal import DiagMatrix
from sympy.matrices.expressions.hadamard import (HadamardPower, HadamardProduct, hadamard_product)
from sympy.matrices.expressions.inverse import Inverse
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import OneMatrix
from sympy.matrices.expressions.trace import Trace
from sympy.matrices.expressions.matadd import MatAdd
from sympy.matrices.expressions.matmul import MatMul
from sympy.matrices.expressions.special import (Identity, ZeroMatrix)
from sympy.tensor.array.array_derivatives import ArrayDerivative
from sympy.matrices.expressions import hadamard_power
from sympy.tensor.array.expressions.array_expressions import ArrayAdd, ArrayTensorProduct, PermuteDims
def test_derivatives_of_complicated_matrix_expr():
    expr = a.T * (A * X * (X.T * B + X * A) + B.T * X.T * (a * b.T * (X * D * X.T + X * (X.T * B + A * X) * D * B - X.T * C.T * A) * B + B * (X * D.T + B * A * X * A.T - 3 * X * D)) * B + 42 * X * B * X.T * A.T * (X + X.T)) * b
    result = (B * (B * A * X * A.T - 3 * X * D + X * D.T) + a * b.T * (X * (A * X + X.T * B) * D * B + X * D * X.T - X.T * C.T * A) * B) * B * b * a.T * B.T + B ** 2 * b * a.T * B.T * X.T * a * b.T * X * D + 42 * A * X * B.T * X.T * a * b.T + B * D * B ** 3 * b * a.T * B.T * X.T * a * b.T * X + B * b * a.T * A * X + a * b.T * (42 * X + 42 * X.T) * A * X * B.T + b * a.T * X * B * a * b.T * B.T ** 2 * X * D.T + b * a.T * X * B * a * b.T * B.T ** 3 * D.T * (B.T * X + X.T * A.T) + 42 * b * a.T * X * B * X.T * A.T + A.T * (42 * X + 42 * X.T) * b * a.T * X * B + A.T * B.T ** 2 * X * B * a * b.T * B.T * A + A.T * a * b.T * (A.T * X.T + B.T * X) + A.T * X.T * b * a.T * X * B * a * b.T * B.T ** 3 * D.T + B.T * X * B * a * b.T * B.T * D - 3 * B.T * X * B * a * b.T * B.T * D.T - C.T * A * B ** 2 * b * a.T * B.T * X.T * a * b.T + X.T * A.T * a * b.T * A.T
    assert expr.diff(X) == result