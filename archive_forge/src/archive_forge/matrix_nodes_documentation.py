from .ast import Token
from sympy.matrices import MatrixExpr
from sympy.core.sympify import sympify
Represents an operation to solve a linear matrix equation.

    Parameters
    ==========

    matrix : MatrixSymbol

      Matrix representing the coefficients of variables in the linear
      equation. This matrix must be square and full-rank (i.e. all columns must
      be linearly independent) for the solving operation to be valid.

    vector : MatrixSymbol

      One-column matrix representing the solutions to the equations
      represented in ``matrix``.

    Examples
    ========

    >>> from sympy import symbols, MatrixSymbol
    >>> from sympy.codegen.matrix_nodes import MatrixSolve
    >>> n = symbols('n', integer=True)
    >>> A = MatrixSymbol('A', n, n)
    >>> x = MatrixSymbol('x', n, 1)
    >>> from sympy.printing.numpy import NumPyPrinter
    >>> NumPyPrinter().doprint(MatrixSolve(A, x))
    'numpy.linalg.solve(A, x)'
    >>> from sympy import octave_code
    >>> octave_code(MatrixSolve(A, x))
    'A \\ x'

    