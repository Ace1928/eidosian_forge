from __future__ import annotations
from functools import wraps
from sympy.core import S, Integer, Basic, Mul, Add
from sympy.core.assumptions import check_assumptions
from sympy.core.decorators import call_highest_priority
from sympy.core.expr import Expr, ExprBuilder
from sympy.core.logic import FuzzyBool
from sympy.core.symbol import Str, Dummy, symbols, Symbol
from sympy.core.sympify import SympifyError, _sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions import conjugate, adjoint
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.common import NonSquareMatrixError
from sympy.matrices.matrices import MatrixKind, MatrixBase
from sympy.multipledispatch import dispatch
from sympy.utilities.misc import filldedent
from .matmul import MatMul
from .matadd import MatAdd
from .matpow import MatPow
from .transpose import Transpose
from .inverse import Inverse
from .special import ZeroMatrix, Identity
from .determinant import Determinant
def contract_one_dims(parts):
    if len(parts) == 1:
        return parts[0]
    else:
        p1, p2 = parts[:2]
        if p2.is_Matrix:
            p2 = p2.T
        if p1 == Identity(1):
            pbase = p2
        elif p2 == Identity(1):
            pbase = p1
        else:
            pbase = p1 * p2
        if len(parts) == 2:
            return pbase
        else:
            if pbase.is_Matrix:
                raise ValueError('')
            return pbase * Mul.fromiter(parts[2:])