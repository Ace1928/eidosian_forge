from sympy.concrete.summations import Sum
from sympy.core.mod import Mod
from sympy.core.relational import (Equality, Unequality)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices.expressions.blockmatrix import BlockMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import Identity
from sympy.utilities.lambdify import lambdify
from sympy.abc import x, i, j, a, b, c, d
from sympy.core import Pow
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2
from sympy.codegen.cfunctions import log1p, expm1, hypot, log10, exp2, log2, Sqrt
from sympy.tensor.array import Array
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, ArrayAdd, \
from sympy.printing.numpy import JaxPrinter, _jax_known_constants, _jax_known_functions
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
from sympy.testing.pytest import skip, raises
from sympy.external import import_module
def test_jax_codegen_extra():
    if not jax:
        skip('JAX not installed')
    M = MatrixSymbol('M', 2, 2)
    N = MatrixSymbol('N', 2, 2)
    P = MatrixSymbol('P', 2, 2)
    Q = MatrixSymbol('Q', 2, 2)
    ma = jax.numpy.array([[1, 2], [3, 4]])
    mb = jax.numpy.array([[1, -2], [-1, 3]])
    mc = jax.numpy.array([[2, 0], [1, 2]])
    md = jax.numpy.array([[1, -1], [4, 7]])
    cg = ArrayTensorProduct(M, N)
    f = lambdify((M, N), cg, 'jax')
    assert (f(ma, mb) == jax.numpy.einsum(ma, [0, 1], mb, [2, 3])).all()
    cg = ArrayAdd(M, N)
    f = lambdify((M, N), cg, 'jax')
    assert (f(ma, mb) == ma + mb).all()
    cg = ArrayAdd(M, N, P)
    f = lambdify((M, N, P), cg, 'jax')
    assert (f(ma, mb, mc) == ma + mb + mc).all()
    cg = ArrayAdd(M, N, P, Q)
    f = lambdify((M, N, P, Q), cg, 'jax')
    assert (f(ma, mb, mc, md) == ma + mb + mc + md).all()
    cg = PermuteDims(M, [1, 0])
    f = lambdify((M,), cg, 'jax')
    assert (f(ma) == ma.T).all()
    cg = PermuteDims(ArrayTensorProduct(M, N), [1, 2, 3, 0])
    f = lambdify((M, N), cg, 'jax')
    assert (f(ma, mb) == jax.numpy.transpose(jax.numpy.einsum(ma, [0, 1], mb, [2, 3]), (1, 2, 3, 0))).all()
    cg = ArrayDiagonal(ArrayTensorProduct(M, N), (1, 2))
    f = lambdify((M, N), cg, 'jax')
    assert (f(ma, mb) == jax.numpy.diagonal(jax.numpy.einsum(ma, [0, 1], mb, [2, 3]), axis1=1, axis2=2)).all()