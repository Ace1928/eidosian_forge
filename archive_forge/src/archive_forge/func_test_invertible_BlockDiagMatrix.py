from sympy.assumptions.ask import (Q, ask)
from sympy.core.symbol import Symbol
from sympy.matrices.expressions.diagonal import (DiagMatrix, DiagonalMatrix)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions import (MatrixSymbol, Identity, ZeroMatrix,
from sympy.matrices.expressions.factorizations import LofLU
from sympy.testing.pytest import XFAIL
def test_invertible_BlockDiagMatrix():
    assert ask(Q.invertible(BlockDiagMatrix(Identity(3), Identity(5)))) == True
    assert ask(Q.invertible(BlockDiagMatrix(ZeroMatrix(3, 3), Identity(5)))) == False
    assert ask(Q.invertible(BlockDiagMatrix(Identity(3), OneMatrix(5, 5)))) == False