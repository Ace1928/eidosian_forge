from sympy.core.numbers import (oo, pi)
from sympy.core.symbol import Wild
from sympy.core.expr import Expr
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import sin, cos, sinc
from sympy.series.series_class import SeriesBase
from sympy.series.sequences import SeqFormula
from sympy.sets.sets import Interval
from sympy.utilities.iterables import is_sequence
def scalex(self, s):
    s, x = (sympify(s), self.x)
    if x in s.free_symbols:
        raise ValueError("'%s' should be independent of %s" % (s, x))
    _expr = self.truncate().subs(x, x * s)
    sfunc = self.function.subs(x, x * s)
    return self.func(sfunc, self.args[1], _expr)