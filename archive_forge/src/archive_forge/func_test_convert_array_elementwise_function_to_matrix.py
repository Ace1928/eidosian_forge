from sympy import Lambda, S, Dummy, KroneckerProduct
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices.expressions.hadamard import HadamardProduct, HadamardPower
from sympy.matrices.expressions.special import (Identity, OneMatrix, ZeroMatrix)
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
from sympy.tensor.array.expressions.from_array_to_matrix import _support_function_tp1_recognize, \
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.combinatorics import Permutation
from sympy.matrices.expressions.diagonal import DiagMatrix, DiagonalMatrix
from sympy.matrices import Trace, MatMul, Transpose
from sympy.tensor.array.expressions.array_expressions import ZeroArray, OneArray, \
from sympy.testing.pytest import raises
def test_convert_array_elementwise_function_to_matrix():
    d = Dummy('d')
    expr = ArrayElementwiseApplyFunc(Lambda(d, sin(d)), x.T * y)
    assert convert_array_to_matrix(expr) == sin(x.T * y)
    expr = ArrayElementwiseApplyFunc(Lambda(d, d ** 2), x.T * y)
    assert convert_array_to_matrix(expr) == (x.T * y) ** 2
    expr = ArrayElementwiseApplyFunc(Lambda(d, sin(d)), x)
    assert convert_array_to_matrix(expr).dummy_eq(x.applyfunc(sin))
    expr = ArrayElementwiseApplyFunc(Lambda(d, 1 / (2 * sqrt(d))), x)
    assert convert_array_to_matrix(expr) == S.Half * HadamardPower(x, -S.Half)