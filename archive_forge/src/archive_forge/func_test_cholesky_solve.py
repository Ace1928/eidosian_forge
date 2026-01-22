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
def test_cholesky_solve():
    A = Matrix([[2, 3, 5], [3, 6, 2], [8, 3, 6]])
    x = Matrix(3, 1, [3, 7, 5])
    b = A * x
    soln = A.cholesky_solve(b)
    assert soln == x
    A = Matrix([[0, -1, 2], [5, 10, 7], [8, 3, 4]])
    x = Matrix(3, 1, [-1, 2, 5])
    b = A * x
    soln = A.cholesky_solve(b)
    assert soln == x
    A = Matrix(((1, 5), (5, 1)))
    x = Matrix((4, -3))
    b = A * x
    soln = A.cholesky_solve(b)
    assert soln == x
    A = Matrix(((9, 3 * I), (-3 * I, 5)))
    x = Matrix((-2, 1))
    b = A * x
    soln = A.cholesky_solve(b)
    assert expand_mul(soln) == x
    A = Matrix(((9 * I, 3), (-3 + I, 5)))
    x = Matrix((2 + 3 * I, -1))
    b = A * x
    soln = A.cholesky_solve(b)
    assert expand_mul(soln) == x
    a00, a01, a11, b0, b1 = symbols('a00, a01, a11, b0, b1')
    A = Matrix(((a00, a01), (a01, a11)))
    b = Matrix((b0, b1))
    x = A.cholesky_solve(b)
    assert simplify(A * x) == b