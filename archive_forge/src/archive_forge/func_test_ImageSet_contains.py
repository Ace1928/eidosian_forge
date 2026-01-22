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
def test_ImageSet_contains():
    assert (2, S.Half) in imageset(x, (x, 1 / x), S.Integers)
    assert imageset(x, x + I * 3, S.Integers).intersection(S.Reals) is S.EmptySet
    i = Dummy(integer=True)
    q = imageset(x, x + I * y, S.Integers).intersection(S.Reals)
    assert q.subs(y, I * i).intersection(S.Integers) is S.Integers
    q = imageset(x, x + I * y / x, S.Integers).intersection(S.Reals)
    assert q.subs(y, 0) is S.Integers
    assert q.subs(y, I * i * x).intersection(S.Integers) is S.Integers
    z = cos(1) ** 2 + sin(1) ** 2 - 1
    q = imageset(x, x + I * z, S.Integers).intersection(S.Reals)
    assert q is not S.EmptySet