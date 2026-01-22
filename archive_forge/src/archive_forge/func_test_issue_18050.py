from sympy.core.expr import unchanged
from sympy.sets.contains import Contains
from sympy.sets.fancysets import (ImageSet, Range, normalize_theta_set,
from sympy.sets.sets import (FiniteSet, Interval, Union, imageset,
from sympy.sets.conditionset import ConditionSet
from sympy.simplify.simplify import simplify
from sympy.core.basic import Basic
from sympy.core.containers import Tuple, TupleKind
from sympy.core.function import Lambda
from sympy.core.kind import NumberKind
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.logic.boolalg import And
from sympy.matrices.dense import eye
from sympy.testing.pytest import XFAIL, raises
from sympy.abc import x, y, t, z
from sympy.core.mod import Mod
import itertools
def test_issue_18050():
    assert imageset(Lambda(x, I * x + 1), S.Integers) == ImageSet(Lambda(x, I * x + 1), S.Integers)
    assert imageset(Lambda(x, 3 * I * x + 4 + 8 * I), S.Integers) == ImageSet(Lambda(x, 3 * I * x + 4 + 2 * I), S.Integers)
    assert imageset(Lambda(x, 2 * x + 3 * I), S.Integers) == ImageSet(Lambda(x, 2 * x + 3 * I), S.Integers)
    r = Symbol('r', positive=True)
    assert imageset(Lambda(x, r * x + 10), S.Integers) == ImageSet(Lambda(x, r * x + 10), S.Integers)
    assert imageset(Lambda(x, 3 * x + 8 + 5 * I), S.Integers) == ImageSet(Lambda(x, 3 * x + 2 + 5 * I), S.Integers)