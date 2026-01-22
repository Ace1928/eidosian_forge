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
def test__dummy_with_inherited_properties_concrete():
    x = Symbol('x')
    from sympy.core.containers import Tuple
    d = _dummy_with_inherited_properties_concrete(Tuple(x, 0, 5))
    assert d.is_real
    assert d.is_integer
    assert d.is_nonnegative
    assert d.is_extended_nonnegative
    d = _dummy_with_inherited_properties_concrete(Tuple(x, 1, 9))
    assert d.is_real
    assert d.is_integer
    assert d.is_positive
    assert d.is_odd is None
    d = _dummy_with_inherited_properties_concrete(Tuple(x, -5, 5))
    assert d.is_real
    assert d.is_integer
    assert d.is_positive is None
    assert d.is_extended_nonnegative is None
    assert d.is_odd is None
    d = _dummy_with_inherited_properties_concrete(Tuple(x, -1.5, 1.5))
    assert d.is_real
    assert d.is_integer is None
    assert d.is_positive is None
    assert d.is_extended_nonnegative is None
    N = Symbol('N', integer=True, positive=True)
    d = _dummy_with_inherited_properties_concrete(Tuple(x, 2, N))
    assert d.is_real
    assert d.is_positive
    assert d.is_integer
    N = Symbol('N', integer=True, positive=True)
    d = _dummy_with_inherited_properties_concrete(Tuple(N, 2, 4))
    assert d is None
    x = Symbol('x', negative=True)
    raises(InconsistentAssumptions, lambda: _dummy_with_inherited_properties_concrete(Tuple(x, 1, 5)))