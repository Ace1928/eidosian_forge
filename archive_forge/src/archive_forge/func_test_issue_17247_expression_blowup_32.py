from sympy.core.function import expand_mul
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.simplify.simplify import simplify
from sympy.matrices.matrices import (ShapeError, NonSquareMatrixError)
from sympy.matrices import (
from sympy.testing.pytest import raises
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.solvers.solveset import linsolve
from sympy.abc import x, y
def test_issue_17247_expression_blowup_32():
    M = Matrix([[x + 1, 1 - x, 0, 0], [1 - x, x + 1, 0, x + 1], [0, 1 - x, x + 1, 0], [0, 0, 0, x + 1]])
    with dotprodsimp(True):
        assert M.LUsolve(ones(4, 1)) == Matrix([[(x + 1) / (4 * x)], [(x - 1) / (4 * x)], [(x + 1) / (4 * x)], [1 / (x + 1)]])