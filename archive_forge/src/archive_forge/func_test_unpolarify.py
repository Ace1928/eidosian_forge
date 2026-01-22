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
def test_unpolarify():
    from sympy.functions.elementary.complexes import polar_lift, principal_branch, unpolarify
    from sympy.core.relational import Ne
    from sympy.functions.elementary.hyperbolic import tanh
    from sympy.functions.special.error_functions import erf
    from sympy.functions.special.gamma_functions import gamma, uppergamma
    from sympy.abc import x
    p = exp_polar(7 * I) + 1
    u = exp(7 * I) + 1
    assert unpolarify(1) == 1
    assert unpolarify(p) == u
    assert unpolarify(p ** 2) == u ** 2
    assert unpolarify(p ** x) == p ** x
    assert unpolarify(p * x) == u * x
    assert unpolarify(p + x) == u + x
    assert unpolarify(sqrt(sin(p))) == sqrt(sin(u))
    t = principal_branch(x, 2 * pi)
    assert unpolarify(t) == x
    assert unpolarify(sqrt(t)) == sqrt(t)
    assert unpolarify(p ** p, exponents_only=True) == p ** u
    assert unpolarify(uppergamma(x, p ** p)) == uppergamma(x, p ** u)
    assert unpolarify(sin(p)) == sin(u)
    assert unpolarify(tanh(p)) == tanh(u)
    assert unpolarify(gamma(p)) == gamma(u)
    assert unpolarify(erf(p)) == erf(u)
    assert unpolarify(uppergamma(x, p)) == uppergamma(x, p)
    assert unpolarify(uppergamma(sin(p), sin(p + exp_polar(0)))) == uppergamma(sin(u), sin(u + 1))
    assert unpolarify(uppergamma(polar_lift(0), 2 * exp_polar(0))) == uppergamma(0, 2)
    assert unpolarify(Eq(p, 0)) == Eq(u, 0)
    assert unpolarify(Ne(p, 0)) == Ne(u, 0)
    assert unpolarify(polar_lift(x) > 0) == (x > 0)
    assert unpolarify(True) is True