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
def test_issue_14640():
    i, n = symbols('i n', integer=True)
    a, b, c = symbols('a b c', zero=False)
    assert Sum(a ** (-i) / (a - b), (i, 0, n)).doit() == Sum(1 / (a * a ** i - a ** i * b), (i, 0, n)).doit() == Piecewise((n + 1, Eq(1 / a, 1)), ((-a ** (-n - 1) + 1) / (1 - 1 / a), True)) / (a - b)
    assert Sum((b * a ** i - c * a ** i) ** (-2), (i, 0, n)).doit() == Piecewise((n + 1, Eq(a ** (-2), 1)), ((-a ** (-2 * n - 2) + 1) / (1 - 1 / a ** 2), True)) / (b - c) ** 2
    s = Sum(i * (a ** (n - i) - b ** (n - i)) / (a - b), (i, 0, n)).doit()
    assert not s.has(Sum)
    assert s.subs({a: 2, b: 3, n: 5}) == 122