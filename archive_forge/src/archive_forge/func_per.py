from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.matrices.common import NonSquareMatrixError
from sympy.assumptions.ask import ask, Q
from sympy.assumptions.refine import handlers_dict
def per(matexpr):
    """ Matrix Permanent

    Examples
    ========

    >>> from sympy import MatrixSymbol, Matrix, per, ones
    >>> A = MatrixSymbol('A', 3, 3)
    >>> per(A)
    Permanent(A)
    >>> per(ones(5, 5))
    120
    >>> M = Matrix([1, 2, 5])
    >>> per(M)
    8
    """
    return Permanent(matexpr).doit()