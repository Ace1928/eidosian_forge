import inspect
import toolz
from toolz.functoolz import (thread_first, thread_last, memoize, curry,
from operator import add, mul, itemgetter
from toolz.utils import raises
from functools import partial
def test_curry_doesnot_transmogrify():

    def f(x, y=0):
        return x + y
    cf = curry(f)
    assert cf(y=1)(y=2)(y=3)(1) == f(1, 3)