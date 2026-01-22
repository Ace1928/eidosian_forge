import inspect
import toolz
from toolz.functoolz import (thread_first, thread_last, memoize, curry,
from operator import add, mul, itemgetter
from toolz.utils import raises
from functools import partial
def test_memoize_curry_cache():

    @memoize(cache={1: True})
    def f(x):
        return False
    assert f(1) is True
    assert f(2) is False