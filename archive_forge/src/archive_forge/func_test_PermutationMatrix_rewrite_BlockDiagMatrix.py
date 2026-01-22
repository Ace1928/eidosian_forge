from sympy.combinatorics import Permutation
from sympy.core.expr import unchanged
from sympy.matrices import Matrix
from sympy.matrices.expressions import \
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import ZeroMatrix, OneMatrix, Identity
from sympy.matrices.expressions.permutation import \
from sympy.testing.pytest import raises
from sympy.core.symbol import Symbol
def test_PermutationMatrix_rewrite_BlockDiagMatrix():
    P = PermutationMatrix(Permutation([0, 1, 2, 3, 4, 5]))
    P0 = PermutationMatrix(Permutation([0]))
    assert P.rewrite(BlockDiagMatrix) == BlockDiagMatrix(P0, P0, P0, P0, P0, P0)
    P = PermutationMatrix(Permutation([0, 1, 3, 2, 4, 5]))
    P10 = PermutationMatrix(Permutation(0, 1))
    assert P.rewrite(BlockDiagMatrix) == BlockDiagMatrix(P0, P0, P10, P0, P0)
    P = PermutationMatrix(Permutation([1, 0, 3, 2, 5, 4]))
    assert P.rewrite(BlockDiagMatrix) == BlockDiagMatrix(P10, P10, P10)
    P = PermutationMatrix(Permutation([0, 4, 3, 2, 1, 5]))
    P3210 = PermutationMatrix(Permutation([3, 2, 1, 0]))
    assert P.rewrite(BlockDiagMatrix) == BlockDiagMatrix(P0, P3210, P0)
    P = PermutationMatrix(Permutation([0, 4, 2, 3, 1, 5]))
    P3120 = PermutationMatrix(Permutation([3, 1, 2, 0]))
    assert P.rewrite(BlockDiagMatrix) == BlockDiagMatrix(P0, P3120, P0)
    P = PermutationMatrix(Permutation(0, 3)(1, 4)(2, 5))
    assert P.rewrite(BlockDiagMatrix) == BlockDiagMatrix(P)