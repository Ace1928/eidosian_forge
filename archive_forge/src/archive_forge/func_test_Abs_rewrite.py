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
def test_Abs_rewrite():
    x = Symbol('x', real=True)
    a = Abs(x).rewrite(Heaviside).expand()
    assert a == x * Heaviside(x) - x * Heaviside(-x)
    for i in [-2, -1, 0, 1, 2]:
        assert a.subs(x, i) == abs(i)
    y = Symbol('y')
    assert Abs(y).rewrite(Heaviside) == Abs(y)
    x, y = (Symbol('x', real=True), Symbol('y'))
    assert Abs(x).rewrite(Piecewise) == Piecewise((x, x >= 0), (-x, True))
    assert Abs(y).rewrite(Piecewise) == Abs(y)
    assert Abs(y).rewrite(sign) == y / sign(y)
    i = Symbol('i', imaginary=True)
    assert abs(i).rewrite(Piecewise) == Piecewise((I * i, I * i >= 0), (-I * i, True))
    assert Abs(y).rewrite(conjugate) == sqrt(y * conjugate(y))
    assert Abs(i).rewrite(conjugate) == sqrt(-i ** 2)
    y = Symbol('y', extended_real=True)
    assert (Abs(exp(-I * x) - exp(-I * y)) ** 2).rewrite(conjugate) == -exp(I * x) * exp(-I * y) + 2 - exp(-I * x) * exp(I * y)