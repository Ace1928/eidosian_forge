from sympy.core.expr import unchanged
from sympy.sets import (ConditionSet, Intersection, FiniteSet,
from sympy.sets.sets import SetKind
from sympy.core.function import (Function, Lambda)
from sympy.core.mod import Mod
from sympy.core.kind import NumberKind
from sympy.core.numbers import (oo, pi)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.trigonometric import (asin, sin)
from sympy.logic.boolalg import And
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.sets.sets import Interval
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_bound_symbols():
    assert ConditionSet(x, Eq(y, 0), FiniteSet(z)).bound_symbols == [x]
    assert ConditionSet(x, Eq(x, 0), FiniteSet(x, y)).bound_symbols == [x]
    assert ConditionSet(x, x < 10, ImageSet(Lambda(y, y ** 2), S.Integers)).bound_symbols == [x]
    assert ConditionSet(x, x < 10, ConditionSet(y, y > 1, S.Integers)).bound_symbols == [x]