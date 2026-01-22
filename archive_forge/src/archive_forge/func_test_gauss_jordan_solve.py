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
def test_gauss_jordan_solve():
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    b = Matrix([3, 6, 9])
    sol, params = A.gauss_jordan_solve(b)
    assert sol == Matrix([[-1], [2], [0]])
    assert params == Matrix(0, 1, [])
    A = eye(3)
    B = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    sol, params = A.gauss_jordan_solve(B)
    assert sol == B
    assert params == Matrix(0, 4, [])
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = Matrix([3, 6, 9])
    sol, params, freevar = A.gauss_jordan_solve(b, freevar=True)
    w = {}
    for s in sol.atoms(Symbol):
        w[s.name] = s
    assert sol == Matrix([[w['tau0'] - 1], [-2 * w['tau0'] + 2], [w['tau0']]])
    assert params == Matrix([[w['tau0']]])
    assert freevar == [2]
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = Matrix([[3, 4], [6, 8], [9, 12]])
    sol, params, freevar = A.gauss_jordan_solve(B, freevar=True)
    w = {}
    for s in sol.atoms(Symbol):
        w[s.name] = s
    assert sol == Matrix([[w['tau0'] - 1, w['tau1'] - Rational(4, 3)], [-2 * w['tau0'] + 2, -2 * w['tau1'] + Rational(8, 3)], [w['tau0'], w['tau1']]])
    assert params == Matrix([[w['tau0'], w['tau1']]])
    assert freevar == [2]
    A = Matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    b = Matrix([0, 0, 0])
    sol, params = A.gauss_jordan_solve(b)
    w = {}
    for s in sol.atoms(Symbol):
        w[s.name] = s
    assert sol == Matrix([[-2 * w['tau0'] - 3 * w['tau1']], [w['tau0']], [w['tau1']]])
    assert params == Matrix([[w['tau0']], [w['tau1']]])
    A = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    b = Matrix([0, 0, 0])
    sol, params = A.gauss_jordan_solve(b)
    w = {}
    for s in sol.atoms(Symbol):
        w[s.name] = s
    assert sol == Matrix([[w['tau0']], [w['tau1']], [w['tau2']]])
    assert params == Matrix([[w['tau0']], [w['tau1']], [w['tau2']]])
    A = Matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    b = Matrix([0, 0, 1])
    raises(ValueError, lambda: A.gauss_jordan_solve(b))
    A = Matrix([[1, 5, 3], [2, 1, 6], [1, 7, 9], [1, 4, 3]])
    b = Matrix([0, 0, 1, 0])
    sol, params = A.gauss_jordan_solve(b)
    assert sol == Matrix([[Rational(-1, 2)], [0], [Rational(1, 6)]])
    assert params == Matrix(0, 1, [])
    A = Matrix([[1, 5, 3], [2, 1, 6], [1, 7, 9], [1, 4, 3]])
    B = Matrix([[0, 0], [0, 0], [1, 2], [0, 0]])
    sol, params = A.gauss_jordan_solve(B)
    assert sol == Matrix([[Rational(-1, 2), Rational(-2, 2)], [0, 0], [Rational(1, 6), Rational(2, 6)]])
    assert params == Matrix(0, 2, [])
    A = Matrix([[1, 5, 3], [2, 1, 6], [1, 7, 9], [1, 4, 3]])
    b = Matrix([0, 0, 0, 1])
    raises(ValueError, lambda: A.gauss_jordan_solve(b))
    A = Matrix([[1, 5, 3], [2, 1, 6], [1, 7, 9], [1, 4, 3]])
    B = Matrix([[0, 0], [0, 0], [1, 0], [0, 1]])
    raises(ValueError, lambda: A.gauss_jordan_solve(B))
    A = Matrix([[1, 5, 3], [2, 1, 6], [1, 7, 9], [1, 4, 3]])
    B = Matrix([[0, 0], [0, 0], [0, 1], [1, 0]])
    raises(ValueError, lambda: A.gauss_jordan_solve(B))
    A = Matrix([[1, 5, 3], [2, 10, 6], [3, 15, 9], [1, 4, 3]])
    b = Matrix([0, 0, 0, 1])
    sol, params = A.gauss_jordan_solve(b)
    w = {}
    for s in sol.atoms(Symbol):
        w[s.name] = s
    assert sol == Matrix([[-3 * w['tau0'] + 5], [-1], [w['tau0']]])
    assert params == Matrix([[w['tau0']]])
    A = Matrix([[1, 5, 3], [2, 10, 6], [3, 15, 9], [1, 4, 3]])
    b = Matrix([0, 0, 1, 1])
    raises(ValueError, lambda: A.gauss_jordan_solve(b))
    A = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 1, 12]])
    b = Matrix([1, 1, 1])
    sol, params = A.gauss_jordan_solve(b)
    w = {}
    for s in sol.atoms(Symbol):
        w[s.name] = s
    assert sol == Matrix([[2 * w['tau0'] - 1], [-3 * w['tau0'] + 1], [0], [w['tau0']]])
    assert params == Matrix([[w['tau0']]])
    A = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [2, 4, 6, 8]])
    b = Matrix([0, 1, 0])
    sol, params = A.gauss_jordan_solve(b)
    w = {}
    for s in sol.atoms(Symbol):
        w[s.name] = s
    assert sol == Matrix([[w['tau0'] + 2 * w['tau1'] + S.Half], [-2 * w['tau0'] - 3 * w['tau1'] - Rational(1, 4)], [w['tau0']], [w['tau1']]])
    assert params == Matrix([[w['tau0']], [w['tau1']]])
    x0, x1, x2, _x0 = symbols('_tau0 _tau1 _tau2 tau1')
    M = Matrix([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, _x0]])
    A = M[:, :-1]
    b = M[:, -1:]
    sol, params = A.gauss_jordan_solve(b)
    assert params == Matrix(3, 1, [x0, x1, x2])
    assert sol == Matrix(5, 1, [x0, 0, x1, _x0, x2])
    A = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [2, 4, 6, 8]])
    b = Matrix([1, 1, 1])
    raises(ValueError, lambda: A.gauss_jordan_solve(b))
    A = ImmutableMatrix([[1, 0], [0, 1]])
    B = ImmutableMatrix([1, 2])
    sol, params = A.gauss_jordan_solve(B)
    assert sol == ImmutableMatrix([1, 2])
    assert params == ImmutableMatrix(0, 1, [])
    assert sol.__class__ == ImmutableDenseMatrix
    assert params.__class__ == ImmutableDenseMatrix
    A = Matrix([[1, 0, 0, 0], [0, 0, 0, 1]])
    b = Matrix([1, 1])
    sol, params = A.gauss_jordan_solve(b)
    w = {}
    for s in sol.atoms(Symbol):
        w[s.name] = s
    assert sol == Matrix([[1], [w['tau0']], [w['tau1']], [1]])
    assert params == Matrix([[w['tau0']], [w['tau1']]])