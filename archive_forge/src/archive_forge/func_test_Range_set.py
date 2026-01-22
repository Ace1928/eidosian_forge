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
def test_Range_set():
    empty = Range(0)
    assert Range(5) == Range(0, 5) == Range(0, 5, 1)
    r = Range(10, 20, 2)
    assert 12 in r
    assert 8 not in r
    assert 11 not in r
    assert 30 not in r
    assert list(Range(0, 5)) == list(range(5))
    assert list(Range(5, 0, -1)) == list(range(5, 0, -1))
    assert Range(5, 15).sup == 14
    assert Range(5, 15).inf == 5
    assert Range(15, 5, -1).sup == 15
    assert Range(15, 5, -1).inf == 6
    assert Range(10, 67, 10).sup == 60
    assert Range(60, 7, -10).inf == 10
    assert len(Range(10, 38, 10)) == 3
    assert Range(0, 0, 5) == empty
    assert Range(oo, oo, 1) == empty
    assert Range(oo, 1, 1) == empty
    assert Range(-oo, 1, -1) == empty
    assert Range(1, oo, -1) == empty
    assert Range(1, -oo, 1) == empty
    assert Range(1, -4, oo) == empty
    ip = symbols('ip', positive=True)
    assert Range(0, ip, -1) == empty
    assert Range(0, -ip, 1) == empty
    assert Range(1, -4, -oo) == Range(1, 2)
    assert Range(1, 4, oo) == Range(1, 2)
    assert Range(-oo, oo).size == oo
    assert Range(oo, -oo, -1).size == oo
    raises(ValueError, lambda: Range(-oo, oo, 2))
    raises(ValueError, lambda: Range(x, pi, y))
    raises(ValueError, lambda: Range(x, y, 0))
    assert 5 in Range(0, oo, 5)
    assert -5 in Range(-oo, 0, 5)
    assert oo not in Range(0, oo)
    ni = symbols('ni', integer=False)
    assert ni not in Range(oo)
    u = symbols('u', integer=None)
    assert Range(oo).contains(u) is not False
    inf = symbols('inf', infinite=True)
    assert inf not in Range(-oo, oo)
    raises(ValueError, lambda: Range(0, oo, 2)[-1])
    raises(ValueError, lambda: Range(0, -oo, -2)[-1])
    assert Range(-oo, 1, 1)[-1] is S.Zero
    assert Range(oo, 1, -1)[-1] == 2
    assert inf not in Range(oo)
    assert Range(1, 10, 1)[-1] == 9
    assert all((i.is_Integer for i in Range(0, -1, 1)))
    it = iter(Range(-oo, 0, 2))
    raises(TypeError, lambda: next(it))
    assert empty.intersect(S.Integers) == empty
    assert Range(-1, 10, 1).intersect(S.Complexes) == Range(-1, 10, 1)
    assert Range(-1, 10, 1).intersect(S.Reals) == Range(-1, 10, 1)
    assert Range(-1, 10, 1).intersect(S.Rationals) == Range(-1, 10, 1)
    assert Range(-1, 10, 1).intersect(S.Integers) == Range(-1, 10, 1)
    assert Range(-1, 10, 1).intersect(S.Naturals) == Range(1, 10, 1)
    assert Range(-1, 10, 1).intersect(S.Naturals0) == Range(0, 10, 1)
    assert Range(1, 10, 1)[5] == 6
    assert Range(1, 12, 2)[5] == 11
    assert Range(1, 10, 1)[-1] == 9
    assert Range(1, 10, 3)[-1] == 7
    raises(ValueError, lambda: Range(oo, 0, -1)[1:3:0])
    raises(ValueError, lambda: Range(oo, 0, -1)[:1])
    raises(ValueError, lambda: Range(1, oo)[-2])
    raises(ValueError, lambda: Range(-oo, 1)[2])
    raises(IndexError, lambda: Range(10)[-20])
    raises(IndexError, lambda: Range(10)[20])
    raises(ValueError, lambda: Range(2, -oo, -2)[2:2:0])
    assert Range(2, -oo, -2)[2:2:2] == empty
    assert Range(2, -oo, -2)[:2:2] == Range(2, -2, -4)
    raises(ValueError, lambda: Range(-oo, 4, 2)[:2:2])
    assert Range(-oo, 4, 2)[::-2] == Range(2, -oo, -4)
    raises(ValueError, lambda: Range(-oo, 4, 2)[::2])
    assert Range(oo, 2, -2)[:] == Range(oo, 2, -2)
    assert Range(-oo, 4, 2)[:-2:-2] == Range(2, 0, -4)
    assert Range(-oo, 4, 2)[:-2:2] == Range(-oo, 0, 4)
    raises(ValueError, lambda: Range(-oo, 4, 2)[:0:-2])
    raises(ValueError, lambda: Range(-oo, 4, 2)[:2:-2])
    assert Range(-oo, 4, 2)[-2::-2] == Range(0, -oo, -4)
    raises(ValueError, lambda: Range(-oo, 4, 2)[-2:0:-2])
    raises(ValueError, lambda: Range(-oo, 4, 2)[0::2])
    assert Range(oo, 2, -2)[0:] == Range(oo, 2, -2)
    raises(ValueError, lambda: Range(-oo, 4, 2)[0:-2:2])
    assert Range(oo, 2, -2)[0:-2] == Range(oo, 6, -2)
    raises(ValueError, lambda: Range(oo, 2, -2)[0:2])
    raises(ValueError, lambda: Range(-oo, 4, 2)[2::-1])
    assert Range(-oo, 4, 2)[-2::2] == Range(0, 4, 4)
    assert Range(oo, 0, -2)[-10:0:2] == empty
    raises(ValueError, lambda: Range(oo, 0, -2)[0])
    raises(ValueError, lambda: Range(oo, 0, -2)[-10:10:2])
    raises(ValueError, lambda: Range(oo, 0, -2)[0::-2])
    assert Range(oo, 0, -2)[0:-4:-2] == empty
    assert Range(oo, 0, -2)[:0:2] == empty
    raises(ValueError, lambda: Range(oo, 0, -2)[:1:-1])
    assert Range(x, x, y) == empty
    assert empty.reversed == empty
    assert 0 not in empty
    assert list(empty) == []
    assert len(empty) == 0
    assert empty.size is S.Zero
    assert empty.intersect(FiniteSet(0)) is S.EmptySet
    assert bool(empty) is False
    raises(IndexError, lambda: empty[0])
    assert empty[:0] == empty
    raises(NotImplementedError, lambda: empty.inf)
    raises(NotImplementedError, lambda: empty.sup)
    assert empty.as_relational(x) is S.false
    AB = [None] + list(range(12))
    for R in [Range(1, 10), Range(1, 10, 2)]:
        r = list(R)
        for a, b, c in itertools.product(AB, AB, [-3, -1, None, 1, 3]):
            for reverse in range(2):
                r = list(reversed(r))
                R = R.reversed
                result = list(R[a:b:c])
                ans = r[a:b:c]
                txt = '\n%s[%s:%s:%s] = %s -> %s' % (R, a, b, c, result, ans)
                check = ans == result
                assert check, txt
    assert Range(1, 10, 1).boundary == Range(1, 10, 1)
    for r in (Range(1, 10, 2), Range(1, oo, 2)):
        rev = r.reversed
        assert r.inf == rev.inf and r.sup == rev.sup
        assert r.step == -rev.step
    builtin_range = range
    raises(TypeError, lambda: Range(builtin_range(1)))
    assert S(builtin_range(10)) == Range(10)
    assert S(builtin_range(1000000000000)) == Range(1000000000000)
    assert Range(1, 4).as_relational(x) == (x >= 1) & (x <= 3) & Eq(Mod(x, 1), 0)
    assert Range(oo, 1, -2).as_relational(x) == (x >= 3) & (x < oo) & Eq(Mod(x + 1, -2), 0)