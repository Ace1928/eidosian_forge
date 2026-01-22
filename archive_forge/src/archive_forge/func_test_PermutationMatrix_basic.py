from sympy.combinatorics import Permutation
from sympy.core.expr import unchanged
from sympy.matrices import Matrix
from sympy.matrices.expressions import \
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import ZeroMatrix, OneMatrix, Identity
from sympy.matrices.expressions.permutation import \
from sympy.testing.pytest import raises
from sympy.core.symbol import Symbol
def test_PermutationMatrix_basic():
    p = Permutation([1, 0])
    assert unchanged(PermutationMatrix, p)
    raises(ValueError, lambda: PermutationMatrix((0, 1, 2)))
    assert PermutationMatrix(p).as_explicit() == Matrix([[0, 1], [1, 0]])
    assert isinstance(PermutationMatrix(p) * MatrixSymbol('A', 2, 2), MatMul)