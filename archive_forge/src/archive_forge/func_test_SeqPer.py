from sympy.core.containers import Tuple
from sympy.core.function import Function
from sympy.core.numbers import oo, Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols, Symbol
from sympy.functions.combinatorial.numbers import tribonacci, fibonacci
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.series import EmptySequence
from sympy.series.sequences import (SeqMul, SeqAdd, SeqPer, SeqFormula,
from sympy.sets.sets import Interval
from sympy.tensor.indexed import Indexed, Idx
from sympy.series.sequences import SeqExpr, SeqExprOp, RecursiveSeq
from sympy.testing.pytest import raises, slow
def test_SeqPer():
    s = SeqPer((1, n, 3), (x, 0, 5))
    assert isinstance(s, SeqPer)
    assert s.periodical == Tuple(1, n, 3)
    assert s.period == 3
    assert s.coeff(3) == 1
    assert s.free_symbols == {n}
    assert list(s) == [1, n, 3, 1, n, 3]
    assert s[:] == [1, n, 3, 1, n, 3]
    assert SeqPer((1, n, 3), (x, -oo, 0))[0:6] == [1, n, 3, 1, n, 3]
    raises(ValueError, lambda: SeqPer((1, 2, 3), (0, 1, 2)))
    raises(ValueError, lambda: SeqPer((1, 2, 3), (x, -oo, oo)))
    raises(ValueError, lambda: SeqPer(n ** 2, (0, oo)))
    assert SeqPer((n, n ** 2, n ** 3), (m, 0, oo))[:6] == [n, n ** 2, n ** 3, n, n ** 2, n ** 3]
    assert SeqPer((n, n ** 2, n ** 3), (n, 0, oo))[:6] == [0, 1, 8, 3, 16, 125]
    assert SeqPer((n, m), (n, 0, oo))[:6] == [0, m, 2, m, 4, m]