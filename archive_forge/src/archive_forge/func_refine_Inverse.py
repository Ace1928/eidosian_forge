from sympy.core.sympify import _sympify
from sympy.core import S, Basic
from sympy.matrices.common import NonSquareMatrixError
from sympy.matrices.expressions.matpow import MatPow
from sympy.assumptions.ask import ask, Q
from sympy.assumptions.refine import handlers_dict
def refine_Inverse(expr, assumptions):
    """
    >>> from sympy import MatrixSymbol, Q, assuming, refine
    >>> X = MatrixSymbol('X', 2, 2)
    >>> X.I
    X**(-1)
    >>> with assuming(Q.orthogonal(X)):
    ...     print(refine(X.I))
    X.T
    """
    if ask(Q.orthogonal(expr), assumptions):
        return expr.arg.T
    elif ask(Q.unitary(expr), assumptions):
        return expr.arg.conjugate()
    elif ask(Q.singular(expr), assumptions):
        raise ValueError('Inverse of singular matrix %s' % expr.arg)
    return expr