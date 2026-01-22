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
def test_Zero_power():
    z1 = ZeroMatrix(n, n)
    assert z1 ** 4 == z1
    raises(ValueError, lambda: z1 ** (-2))
    assert z1 ** 0 == Identity(n)
    assert MatPow(z1, 2).doit() == z1 ** 2
    raises(ValueError, lambda: MatPow(z1, -2).doit())
    z2 = ZeroMatrix(3, 3)
    assert MatPow(z2, 4).doit() == z2 ** 4
    raises(ValueError, lambda: z2 ** (-3))
    assert z2 ** 3 == MatPow(z2, 3).doit()
    assert z2 ** 0 == Identity(3)
    raises(ValueError, lambda: MatPow(z2, -1).doit())