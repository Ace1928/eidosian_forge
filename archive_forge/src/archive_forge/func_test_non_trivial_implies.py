from sympy.assumptions.ask import (Q, ask)
from sympy.core.symbol import Symbol
from sympy.matrices.expressions.diagonal import (DiagMatrix, DiagonalMatrix)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions import (MatrixSymbol, Identity, ZeroMatrix,
from sympy.matrices.expressions.factorizations import LofLU
from sympy.testing.pytest import XFAIL
@XFAIL
def test_non_trivial_implies():
    X = MatrixSymbol('X', 3, 3)
    Y = MatrixSymbol('Y', 3, 3)
    assert ask(Q.lower_triangular(X + Y), Q.lower_triangular(X) & Q.lower_triangular(Y)) is True
    assert ask(Q.triangular(X), Q.lower_triangular(X)) is True
    assert ask(Q.triangular(X + Y), Q.lower_triangular(X) & Q.lower_triangular(Y)) is True