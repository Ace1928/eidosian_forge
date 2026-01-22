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
def test_evalf_euler_maclaurin():
    assert NS(Sum(1 / k ** k, (k, 1, oo)), 15) == '1.29128599706266'
    assert NS(Sum(1 / k ** k, (k, 1, oo)), 50) == '1.2912859970626635404072825905956005414986193682745'
    assert NS(Sum(1 / k - log(1 + 1 / k), (k, 1, oo)), 15) == NS(EulerGamma, 15)
    assert NS(Sum(1 / k - log(1 + 1 / k), (k, 1, oo)), 50) == NS(EulerGamma, 50)
    assert NS(Sum(log(k) / k ** 2, (k, 1, oo)), 15) == '0.937548254315844'
    assert NS(Sum(log(k) / k ** 2, (k, 1, oo)), 50) == '0.93754825431584375370257409456786497789786028861483'
    assert NS(Sum(1 / k, (k, 1000000, 2000000)), 15) == '0.693147930560008'
    assert NS(Sum(1 / k, (k, 1000000, 2000000)), 50) == '0.69314793056000780941723211364567656807940638436025'