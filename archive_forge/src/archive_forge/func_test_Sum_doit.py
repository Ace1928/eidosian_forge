from math import prod
from sympy.concrete.expr_with_intlimits import ReorderError
from sympy.concrete.products import (Product, product)
from sympy.concrete.summations import (Sum, summation, telescopic,
from sympy.core.function import (Derivative, Function)
from sympy.core import (Catalan, EulerGamma)
from sympy.core.facts import InconsistentAssumptions
from sympy.core.mod import Mod
from sympy.core.numbers import (E, I, Rational, nan, oo, pi)
from sympy.core.relational import Eq
from sympy.core.numbers import Float
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (rf, binomial, factorial)
from sympy.functions.combinatorial.numbers import harmonic
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (sinh, tanh)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.gamma_functions import (gamma, lowergamma)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.functions.special.zeta_functions import zeta
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And, Or
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import Identity
from sympy.matrices import (Matrix, SparseMatrix,
from sympy.sets.fancysets import Range
from sympy.sets.sets import Interval
from sympy.simplify.combsimp import combsimp
from sympy.simplify.simplify import simplify
from sympy.tensor.indexed import (Idx, Indexed, IndexedBase)
from sympy.testing.pytest import XFAIL, raises, slow
from sympy.abc import a, b, c, d, k, m, x, y, z
def test_Sum_doit():
    assert Sum(n * Integral(a ** 2), (n, 0, 2)).doit() == a ** 3
    assert Sum(n * Integral(a ** 2), (n, 0, 2)).doit(deep=False) == 3 * Integral(a ** 2)
    assert summation(n * Integral(a ** 2), (n, 0, 2)) == 3 * Integral(a ** 2)
    s = Sum(Sum(Sum(2, (z, 1, n + 1)), (y, x + 1, n)), (x, 1, n))
    assert 0 == (s.doit() - n * (n + 1) * (n - 1)).factor()
    assert Sum(KroneckerDelta(x, y), (x, -oo, oo)).doit() == Piecewise((1, And(-oo < y, y < oo)), (0, True))
    assert Sum(KroneckerDelta(m, n), (m, -oo, oo)).doit() == 1
    assert Sum(m * KroneckerDelta(x, y), (x, -oo, oo)).doit() == Piecewise((m, And(-oo < y, y < oo)), (0, True))
    assert Sum(x * KroneckerDelta(m, n), (m, -oo, oo)).doit() == x
    assert Sum(Sum(KroneckerDelta(m, n), (m, 1, 3)), (n, 1, 3)).doit() == 3
    assert Sum(Sum(KroneckerDelta(k, m), (m, 1, 3)), (n, 1, 3)).doit() == 3 * Piecewise((1, And(1 <= k, k <= 3)), (0, True))
    assert Sum(f(n) * Sum(KroneckerDelta(m, n), (m, 0, oo)), (n, 1, 3)).doit() == f(1) + f(2) + f(3)
    assert Sum(f(n) * Sum(KroneckerDelta(m, n), (m, 0, oo)), (n, 1, oo)).doit() == Sum(f(n), (n, 1, oo))
    nmax = symbols('N', integer=True, positive=True)
    pw = Piecewise((1, And(1 <= n, n <= nmax)), (0, True))
    assert Sum(pw, (n, 1, nmax)).doit() == Sum(Piecewise((1, nmax >= n), (0, True)), (n, 1, nmax))
    q, s = symbols('q, s')
    assert summation(1 / n ** (2 * s), (n, 1, oo)) == Piecewise((zeta(2 * s), 2 * s > 1), (Sum(n ** (-2 * s), (n, 1, oo)), True))
    assert summation(1 / (n + 1) ** s, (n, 0, oo)) == Piecewise((zeta(s), s > 1), (Sum((n + 1) ** (-s), (n, 0, oo)), True))
    assert summation(1 / (n + q) ** s, (n, 0, oo)) == Piecewise((zeta(s, q), And(q > 0, s > 1)), (Sum((n + q) ** (-s), (n, 0, oo)), True))
    assert summation(1 / (n + q) ** s, (n, q, oo)) == Piecewise((zeta(s, 2 * q), And(2 * q > 0, s > 1)), (Sum((n + q) ** (-s), (n, q, oo)), True))
    assert summation(1 / n ** 2, (n, 1, oo)) == zeta(2)
    assert summation(1 / n ** s, (n, 0, oo)) == Sum(n ** (-s), (n, 0, oo))