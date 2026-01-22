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
def test_hypersum():
    assert simplify(summation(x ** n / fac(n), (n, 1, oo))) == -1 + exp(x)
    assert summation((-1) ** n * x ** (2 * n) / fac(2 * n), (n, 0, oo)) == cos(x)
    assert simplify(summation((-1) ** n * x ** (2 * n + 1) / factorial(2 * n + 1), (n, 3, oo))) == -x + sin(x) + x ** 3 / 6 - x ** 5 / 120
    assert summation(1 / (n + 2) ** 3, (n, 1, oo)) == Rational(-9, 8) + zeta(3)
    assert summation(1 / n ** 4, (n, 1, oo)) == pi ** 4 / 90
    s = summation(x ** n * n, (n, -oo, 0))
    assert s.is_Piecewise
    assert s.args[0].args[0] == -1 / (x * (1 - 1 / x) ** 2)
    assert s.args[0].args[1] == (abs(1 / x) < 1)
    m = Symbol('n', integer=True, positive=True)
    assert summation(binomial(m, k), (k, 0, m)) == 2 ** m