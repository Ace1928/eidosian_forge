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
@slow
def test_is_convergent():
    assert Sum(n / (2 * n + 1), (n, 1, oo)).is_convergent() is S.false
    assert Sum(factorial(n) / 5 ** n, (n, 1, oo)).is_convergent() is S.false
    assert Sum(3 ** (-2 * n - 1) * n ** n, (n, 1, oo)).is_convergent() is S.false
    assert Sum((-1) ** n * n, (n, 3, oo)).is_convergent() is S.false
    assert Sum((-1) ** n, (n, 1, oo)).is_convergent() is S.false
    assert Sum(log(1 / n), (n, 2, oo)).is_convergent() is S.false
    assert Sum(Product(3 * m, (m, 1, n)) / Product(3 * m + 4, (m, 1, n)), (n, 1, oo)).is_convergent() is S.true
    assert Sum((-12) ** n / n, (n, 1, oo)).is_convergent() is S.false
    assert Sum(1 / (n ** 2 + 1), (n, 1, oo)).is_convergent() is S.true
    assert Sum(1 / n ** Rational(6, 5), (n, 1, oo)).is_convergent() is S.true
    assert Sum(2 / (n * sqrt(n - 1)), (n, 2, oo)).is_convergent() is S.true
    assert Sum(1 / (sqrt(n) * sqrt(n)), (n, 2, oo)).is_convergent() is S.false
    assert Sum(factorial(n) / factorial(n + 2), (n, 1, oo)).is_convergent() is S.true
    assert Sum(rf(5, n) / rf(7, n), (n, 1, oo)).is_convergent() is S.true
    assert Sum(rf(1, n) * rf(2, n) / (rf(3, n) * factorial(n)), (n, 1, oo)).is_convergent() is S.false
    assert Sum(1 / (n + log(n)), (n, 1, oo)).is_convergent() is S.false
    assert Sum(1 / (n ** 2 * log(n)), (n, 2, oo)).is_convergent() is S.true
    assert Sum(1 / (n * log(n)), (n, 2, oo)).is_convergent() is S.false
    assert Sum(2 / (n * log(n) * log(log(n)) ** 2), (n, 5, oo)).is_convergent() is S.true
    assert Sum(2 / (n * log(n) ** 2), (n, 2, oo)).is_convergent() is S.true
    assert Sum((n - 1) / (n ** 2 * log(n) ** 3), (n, 2, oo)).is_convergent() is S.true
    assert Sum(1 / (n * log(n) * log(log(n))), (n, 5, oo)).is_convergent() is S.false
    assert Sum((n - 1) / (n * log(n) ** 3), (n, 3, oo)).is_convergent() is S.false
    assert Sum(2 / (n ** 2 * log(n)), (n, 2, oo)).is_convergent() is S.true
    assert Sum(1 / (n * sqrt(log(n)) * log(log(n))), (n, 100, oo)).is_convergent() is S.false
    assert Sum(log(log(n)) / (n * log(n) ** 2), (n, 100, oo)).is_convergent() is S.true
    assert Sum(log(n) / n ** 2, (n, 5, oo)).is_convergent() is S.true
    assert Sum((-1) ** (n - 1) / (n ** 2 - 1), (n, 3, oo)).is_convergent() is S.true
    assert Sum(1 / (n ** 2 + 1), (n, -oo, 1)).is_convergent() is S.true
    assert Sum(1 / (n - 1), (n, -oo, -1)).is_convergent() is S.false
    assert Sum(1 / (n ** 2 - 1), (n, -oo, -5)).is_convergent() is S.true
    assert Sum(1 / (n ** 2 - 1), (n, -oo, 2)).is_convergent() is S.true
    assert Sum(1 / (n ** 2 - 1), (n, -oo, oo)).is_convergent() is S.true
    f = Piecewise((n ** (-2), n <= 1), (n ** 2, n > 1))
    assert Sum(f, (n, 1, oo)).is_convergent() is S.false
    assert Sum(f, (n, -oo, oo)).is_convergent() is S.false
    assert Sum(f, (n, 1, 100)).is_convergent() is S.true
    assert Sum(log(n) / n ** 3, (n, 1, oo)).is_convergent() is S.true
    assert Sum(-log(n) / n ** 3, (n, 1, oo)).is_convergent() is S.true
    eq = (x - 2) * (x ** 2 - 6 * x + 4) * exp(-x)
    assert Sum(eq, (x, 1, oo)).is_convergent() is S.true
    assert Sum(eq, (x, 1, 2)).is_convergent() is S.true
    assert Sum(1 / x ** 3, (x, 1, oo)).is_convergent() is S.true
    assert Sum(1 / x ** S.Half, (x, 1, oo)).is_convergent() is S.false
    assert Sum(1 / n - 3 / (3 * n + 2), (n, 1, oo)).is_convergent() is S.true
    assert Sum(4 / (n + 2) - 5 / (n + 1) + 1 / n, (n, 7, oo)).is_convergent() is S.true