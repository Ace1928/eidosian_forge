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
def test_MatrixSet():
    M = MatrixSet(2, 2, set=S.Reals)
    assert M.shape == (2, 2)
    assert M.set == S.Reals
    X = Matrix([[1, 2], [3, 4]])
    assert X in M
    X = ZeroMatrix(2, 2)
    assert X in M
    raises(TypeError, lambda: A in M)
    raises(TypeError, lambda: 1 in M)
    M = MatrixSet(n, m, set=S.Reals)
    assert A in M
    raises(TypeError, lambda: C in M)
    raises(TypeError, lambda: X in M)
    M = MatrixSet(2, 2, set={1, 2, 3})
    X = Matrix([[1, 2], [3, 4]])
    Y = Matrix([[1, 2]])
    assert (X in M) == S.false
    assert (Y in M) == S.false
    raises(ValueError, lambda: MatrixSet(2, -2, S.Reals))
    raises(ValueError, lambda: MatrixSet(2.4, -1, S.Reals))
    raises(TypeError, lambda: MatrixSet(2, 2, (1, 2, 3)))