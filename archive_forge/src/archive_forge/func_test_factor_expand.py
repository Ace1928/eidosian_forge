from sympy.concrete.summations import Sum
from sympy.core.exprtools import gcd_terms
from sympy.core.function import (diff, expand)
from sympy.core.relational import Eq
from sympy.core.symbol import (Dummy, Symbol, Str)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import zeros
from sympy.polys.polytools import factor
from sympy.core import (S, symbols, Add, Mul, SympifyError, Rational,
from sympy.functions import sin, cos, tan, sqrt, cbrt, exp
from sympy.simplify import simplify
from sympy.matrices import (ImmutableMatrix, Inverse, MatAdd, MatMul,
from sympy.matrices.common import NonSquareMatrixError
from sympy.matrices.expressions.determinant import Determinant, det
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.matrices.expressions.special import ZeroMatrix, Identity
from sympy.testing.pytest import raises, XFAIL
@XFAIL
def test_factor_expand():
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, n)
    expr1 = (A + B) * (C + D)
    expr2 = A * C + B * C + A * D + B * D
    assert expr1 != expr2
    assert expand(expr1) == expr2
    assert factor(expr2) == expr1
    expr = B ** (-1) * (A ** (-1) * B ** (-1) - A ** (-1) * C * B ** (-1)) ** (-1) * A ** (-1)
    I = Identity(n)
    assert factor(expr) in [I - C, B ** (-1) * (A ** (-1) * (I - C) * B ** (-1)) ** (-1) * A ** (-1)]