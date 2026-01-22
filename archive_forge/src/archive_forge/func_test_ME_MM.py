from sympy.core.symbol import symbols
from sympy.matrices import (Matrix, MatrixSymbol, eye, Identity,
from sympy.matrices.expressions import MatrixExpr, MatAdd
from sympy.matrices.common import classof
from sympy.testing.pytest import raises
def test_ME_MM():
    assert isinstance(Identity(3) + MM, MatrixExpr)
    assert isinstance(SM + MM, MatAdd)
    assert isinstance(MM + SM, MatAdd)
    assert (Identity(3) + MM)[1, 1] == 6