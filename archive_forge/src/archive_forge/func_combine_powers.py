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
def combine_powers(mul):
    """Combine consecutive powers with the same base into one, e.g.
    $$A \\times A^2 \\Rightarrow A^3$$

    This also cancels out the possible matrix inverses using the
    knowledgebase of :class:`~.Inverse`, e.g.,
    $$ Y \\times X \\times X^{-1} \\Rightarrow Y $$
    """
    factor, args = mul.as_coeff_matrices()
    new_args = [args[0]]
    for i in range(1, len(args)):
        A = new_args[-1]
        B = args[i]
        if isinstance(B, Inverse) and isinstance(B.arg, MatMul):
            Bargs = B.arg.args
            l = len(Bargs)
            if list(Bargs) == new_args[-l:]:
                new_args = new_args[:-l] + [Identity(B.shape[0])]
                continue
        if isinstance(A, Inverse) and isinstance(A.arg, MatMul):
            Aargs = A.arg.args
            l = len(Aargs)
            if list(Aargs) == args[i:i + l]:
                identity = Identity(A.shape[0])
                new_args[-1] = identity
                for j in range(i, i + l):
                    args[j] = identity
                continue
        if A.is_square == False or B.is_square == False:
            new_args.append(B)
            continue
        if isinstance(A, MatPow):
            A_base, A_exp = A.args
        else:
            A_base, A_exp = (A, S.One)
        if isinstance(B, MatPow):
            B_base, B_exp = B.args
        else:
            B_base, B_exp = (B, S.One)
        if A_base == B_base:
            new_exp = A_exp + B_exp
            new_args[-1] = MatPow(A_base, new_exp).doit(deep=False)
            continue
        elif not isinstance(B_base, MatrixBase):
            try:
                B_base_inv = B_base.inverse()
            except NonInvertibleMatrixError:
                B_base_inv = None
            if B_base_inv is not None and A_base == B_base_inv:
                new_exp = A_exp - B_exp
                new_args[-1] = MatPow(A_base, new_exp).doit(deep=False)
                continue
        new_args.append(B)
    return newmul(factor, *new_args)