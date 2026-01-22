from sympy.assumptions.ask import ask, Q
from sympy.assumptions.refine import handlers_dict
from sympy.core import Basic, sympify, S
from sympy.core.mul import mul, Mul
from sympy.core.numbers import Number, Integer
from sympy.core.symbol import Dummy
from sympy.functions import adjoint
from sympy.strategies import (rm_id, unpack, typed, flatten, exhaust,
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices.matrices import MatrixBase
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.matrices.expressions._shape import validate_matmul_integer as validate
from .inverse import Inverse
from .matexpr import MatrixExpr
from .matpow import MatPow
from .transpose import transpose
from .permutation import PermutationMatrix
from .special import ZeroMatrix, Identity, GenericIdentity, OneMatrix
def only_squares(*matrices):
    """factor matrices only if they are square"""
    if matrices[0].rows != matrices[-1].cols:
        raise RuntimeError('Invalid matrices being multiplied')
    out = []
    start = 0
    for i, M in enumerate(matrices):
        if M.cols == matrices[start].rows:
            out.append(MatMul(*matrices[start:i + 1]).doit())
            start = i + 1
    return out