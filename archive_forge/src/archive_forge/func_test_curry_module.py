import inspect
import toolz
from toolz.functoolz import (thread_first, thread_last, memoize, curry,
from operator import add, mul, itemgetter
from toolz.utils import raises
from functools import partial
def test_curry_module():
    from toolz.curried.exceptions import merge
    assert merge.__module__ == 'toolz.curried.exceptions'