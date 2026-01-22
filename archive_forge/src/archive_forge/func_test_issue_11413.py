from sympy.core.expr import Expr
from sympy.core.function import (Derivative, Function, Lambda, expand)
from sympy.core.numbers import (E, I, Rational, comp, nan, oo, pi, zoo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (Abs, adjoint, arg, conjugate, im, re, sign, transpose)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, atan, atan2, cos, sin)
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.integrals.integrals import Integral
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.funcmatrix import FunctionMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.immutable import (ImmutableMatrix, ImmutableSparseMatrix)
from sympy.matrices import SparseMatrix
from sympy.sets.sets import Interval
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.testing.pytest import XFAIL, raises, _both_exp_pow
def test_issue_11413():
    from sympy.simplify.simplify import simplify
    v0 = Symbol('v0')
    v1 = Symbol('v1')
    v2 = Symbol('v2')
    V = Matrix([[v0], [v1], [v2]])
    U = V.normalized()
    assert U == Matrix([[v0 / sqrt(Abs(v0) ** 2 + Abs(v1) ** 2 + Abs(v2) ** 2)], [v1 / sqrt(Abs(v0) ** 2 + Abs(v1) ** 2 + Abs(v2) ** 2)], [v2 / sqrt(Abs(v0) ** 2 + Abs(v1) ** 2 + Abs(v2) ** 2)]])
    U.norm = sqrt(v0 ** 2 / (v0 ** 2 + v1 ** 2 + v2 ** 2) + v1 ** 2 / (v0 ** 2 + v1 ** 2 + v2 ** 2) + v2 ** 2 / (v0 ** 2 + v1 ** 2 + v2 ** 2))
    assert simplify(U.norm) == 1