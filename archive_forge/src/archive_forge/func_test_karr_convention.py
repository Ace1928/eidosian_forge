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
def test_karr_convention():
    i = Symbol('i', integer=True)
    k = Symbol('k', integer=True)
    j = Symbol('j', integer=True)
    m = k
    n = k + j
    a = m
    b = n - 1
    S1 = Sum(i ** 2, (i, a, b)).doit()
    m = k + j
    n = k
    a = m
    b = n - 1
    S2 = Sum(i ** 2, (i, a, b)).doit()
    assert simplify(S1 + S2) == 0
    m = k
    n = k
    a = m
    b = n - 1
    Sz = Sum(i ** 2, (i, a, b)).doit()
    assert Sz == 0
    m = 2
    n = 11
    a = m
    b = n - 1
    S1 = Sum(f(i), (i, a, b)).doit()
    m = 11
    n = 2
    a = m
    b = n - 1
    S2 = Sum(f(i), (i, a, b)).doit()
    assert simplify(S1 + S2) == 0
    m = 5
    n = 5
    a = m
    b = n - 1
    Sz = Sum(f(i), (i, a, b)).doit()
    assert Sz == 0
    e = Piecewise((exp(-i), Mod(i, 2) > 0), (0, True))
    s = Sum(e, (i, 0, 11))
    assert s.n(3) == s.doit().n(3)