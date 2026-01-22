from sympy.core.sympify import _sympify
from sympy.matrices.expressions import MatrixExpr
from sympy.core import S, Eq, Ge
from sympy.core.mul import Mul
from sympy.functions.special.tensor_functions import KroneckerDelta
@property
def diagonal_length(self):
    return self.shape[0]