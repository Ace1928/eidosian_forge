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
def test_derivatives_of_hadamard_expressions():
    expr = hadamard_product(a, x, b)
    assert expr.diff(x) == DiagMatrix(hadamard_product(b, a))
    expr = a.T * hadamard_product(A, X, B) * b
    assert expr.diff(X) == HadamardProduct(a * b.T, A, B)
    expr = hadamard_power(x, 2)
    assert expr.diff(x).doit() == 2 * DiagMatrix(x)
    expr = hadamard_power(x.T, 2)
    assert expr.diff(x).doit() == 2 * DiagMatrix(x)
    expr = hadamard_power(x, S.Half)
    assert expr.diff(x) == S.Half * DiagMatrix(hadamard_power(x, Rational(-1, 2)))
    expr = hadamard_power(a.T * X * b, 2)
    assert expr.diff(X) == 2 * a * a.T * X * b * b.T
    expr = hadamard_power(a.T * X * b, S.Half)
    assert expr.diff(X) == a / (2 * sqrt(a.T * X * b)) * b.T