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
def test_arithmetic_sums():
    assert summation(1, (n, a, b)) == b - a + 1
    assert Sum(S.NaN, (n, a, b)) is S.NaN
    assert Sum(x, (n, a, a)).doit() == x
    assert Sum(x, (x, a, a)).doit() == a
    assert Sum(x, (n, 1, a)).doit() == a * x
    assert Sum(x, (x, Range(1, 11))).doit() == 55
    assert Sum(x, (x, Range(1, 11, 2))).doit() == 25
    assert Sum(x, (x, Range(1, 10, 2))) == Sum(x, (x, Range(9, 0, -2)))
    lo, hi = (1, 2)
    s1 = Sum(n, (n, lo, hi))
    s2 = Sum(n, (n, hi, lo))
    assert s1 != s2
    assert s1.doit() == 3 and s2.doit() == 0
    lo, hi = (x, x + 1)
    s1 = Sum(n, (n, lo, hi))
    s2 = Sum(n, (n, hi, lo))
    assert s1 != s2
    assert s1.doit() == 2 * x + 1 and s2.doit() == 0
    assert Sum(Integral(x, (x, 1, y)) + x, (x, 1, 2)).doit() == y ** 2 + 2
    assert summation(1, (n, 1, 10)) == 10
    assert summation(2 * n, (n, 0, 10 ** 10)) == 100000000010000000000
    assert summation(4 * n * m, (n, a, 1), (m, 1, d)).expand() == 2 * d + 2 * d ** 2 + a * d + a * d ** 2 - d * a ** 2 - a ** 2 * d ** 2
    assert summation(cos(n), (n, -2, 1)) == cos(-2) + cos(-1) + cos(0) + cos(1)
    assert summation(cos(n), (n, x, x + 2)) == cos(x) + cos(x + 1) + cos(x + 2)
    assert isinstance(summation(cos(n), (n, x, x + S.Half)), Sum)
    assert summation(k, (k, 0, oo)) is oo
    assert summation(k, (k, Range(1, 11))) == 55