from sympy.assumptions.ask import (Q, ask)
from sympy.core.symbol import Symbol
from sympy.matrices.expressions.diagonal import (DiagMatrix, DiagonalMatrix)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions import (MatrixSymbol, Identity, ZeroMatrix,
from sympy.matrices.expressions.factorizations import LofLU
from sympy.testing.pytest import XFAIL
def test_field_assumptions():
    X = MatrixSymbol('X', 4, 4)
    Y = MatrixSymbol('Y', 4, 4)
    assert ask(Q.real_elements(X), Q.real_elements(X))
    assert not ask(Q.integer_elements(X), Q.real_elements(X))
    assert ask(Q.complex_elements(X), Q.real_elements(X))
    assert ask(Q.complex_elements(X ** 2), Q.real_elements(X))
    assert ask(Q.real_elements(X ** 2), Q.integer_elements(X))
    assert ask(Q.real_elements(X + Y), Q.real_elements(X)) is None
    assert ask(Q.real_elements(X + Y), Q.real_elements(X) & Q.real_elements(Y))
    from sympy.matrices.expressions.hadamard import HadamardProduct
    assert ask(Q.real_elements(HadamardProduct(X, Y)), Q.real_elements(X) & Q.real_elements(Y))
    assert ask(Q.complex_elements(X + Y), Q.real_elements(X) & Q.complex_elements(Y))
    assert ask(Q.real_elements(X.T), Q.real_elements(X))
    assert ask(Q.real_elements(X.I), Q.real_elements(X) & Q.invertible(X))
    assert ask(Q.real_elements(Trace(X)), Q.real_elements(X))
    assert ask(Q.integer_elements(Determinant(X)), Q.integer_elements(X))
    assert not ask(Q.integer_elements(X.I), Q.integer_elements(X))
    alpha = Symbol('alpha')
    assert ask(Q.real_elements(alpha * X), Q.real_elements(X) & Q.real(alpha))
    assert ask(Q.real_elements(LofLU(X)), Q.real_elements(X))
    e = Symbol('e', integer=True, negative=True)
    assert ask(Q.real_elements(X ** e), Q.real_elements(X) & Q.invertible(X))
    assert ask(Q.real_elements(X ** e), Q.real_elements(X)) is None