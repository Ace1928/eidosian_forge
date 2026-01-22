from sympy.core.symbol import symbols, Dummy
from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
from sympy.core.function import Lambda
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.matmul import MatMul
from sympy.simplify.simplify import simplify
def test_applyfunc_shape_11_matrices():
    M = MatrixSymbol('M', 1, 1)
    double = Lambda(x, x * 2)
    expr = M.applyfunc(sin)
    assert isinstance(expr, ElementwiseApplyFunction)
    expr = M.applyfunc(double)
    assert isinstance(expr, MatMul)
    assert expr == 2 * M