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
def test_process_limits():
    from sympy.concrete.expr_with_limits import _process_limits
    raises(ValueError, lambda: _process_limits(Range(3), discrete=True))
    raises(ValueError, lambda: _process_limits(Range(3), discrete=False))
    union = Or(x < 1, x > 3).as_set()
    raises(ValueError, lambda: _process_limits(union, discrete=True))
    raises(ValueError, lambda: _process_limits(union, discrete=False))
    assert _process_limits((x, 1, 2)) == ([(x, 1, 2)], 1)
    assert isinstance(S.Reals, Interval)
    C = Integral
    assert C(x, x >= 5) == C(x, (x, 5, oo))
    assert C(x, x < 3) == C(x, (x, -oo, 3))
    ans = C(x, (x, 0, 3))
    assert C(x, And(x >= 0, x < 3)) == ans
    assert C(x, (x, Interval.Ropen(0, 3))) == ans
    raises(TypeError, lambda: C(x, (x, Range(3))))
    for D in (Sum, Product):
        r, ans = (Range(3, 10, 2), D(2 * x + 3, (x, 0, 3)))
        assert D(x, (x, r)) == ans
        assert D(x, (x, r.reversed)) == ans
        r, ans = (Range(3, oo, 2), D(2 * x + 3, (x, 0, oo)))
        assert D(x, (x, r)) == ans
        assert D(x, (x, r.reversed)) == ans
        r, ans = (Range(-oo, 5, 2), D(3 - 2 * x, (x, 0, oo)))
        assert D(x, (x, r)) == ans
        assert D(x, (x, r.reversed)) == ans
        raises(TypeError, lambda: D(x, x > 0))
        raises(ValueError, lambda: D(x, Interval(1, 3)))
        raises(NotImplementedError, lambda: D(x, (x, union)))