from sympy.assumptions.ask import (Q, ask)
from sympy.core.symbol import Symbol
from sympy.matrices.expressions.diagonal import (DiagMatrix, DiagonalMatrix)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions import (MatrixSymbol, Identity, ZeroMatrix,
from sympy.matrices.expressions.factorizations import LofLU
from sympy.testing.pytest import XFAIL
def test_fullrank():
    assert ask(Q.fullrank(X), Q.fullrank(X))
    assert ask(Q.fullrank(X ** 2), Q.fullrank(X))
    assert ask(Q.fullrank(X.T), Q.fullrank(X)) is True
    assert ask(Q.fullrank(X)) is None
    assert ask(Q.fullrank(Y)) is None
    assert ask(Q.fullrank(X * Z), Q.fullrank(X) & Q.fullrank(Z)) is True
    assert ask(Q.fullrank(Identity(3))) is True
    assert ask(Q.fullrank(ZeroMatrix(3, 3))) is False
    assert ask(Q.fullrank(OneMatrix(1, 1))) is True
    assert ask(Q.fullrank(OneMatrix(3, 3))) is False
    assert ask(Q.invertible(X), ~Q.fullrank(X)) == False