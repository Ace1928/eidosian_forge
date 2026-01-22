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
def combine_permutations(mul):
    """Refine products of permutation matrices as the products of cycles.
    """
    args = mul.args
    l = len(args)
    if l < 2:
        return mul
    result = [args[0]]
    for i in range(1, l):
        A = result[-1]
        B = args[i]
        if isinstance(A, PermutationMatrix) and isinstance(B, PermutationMatrix):
            cycle_1 = A.args[0]
            cycle_2 = B.args[0]
            result[-1] = PermutationMatrix(cycle_1 * cycle_2)
        else:
            result.append(B)
    return MatMul(*result)