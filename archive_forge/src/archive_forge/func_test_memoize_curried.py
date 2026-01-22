import inspect
import toolz
from toolz.functoolz import (thread_first, thread_last, memoize, curry,
from operator import add, mul, itemgetter
from toolz.utils import raises
from functools import partial
def test_memoize_curried():

    @curry
    def f(x, y=0):
        return x + y
    f2 = f(y=1)
    fm2 = memoize(f2)
    assert fm2(3) == f2(3)
    assert fm2(3) == f2(3)