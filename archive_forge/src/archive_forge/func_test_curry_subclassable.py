import inspect
import toolz
from toolz.functoolz import (thread_first, thread_last, memoize, curry,
from operator import add, mul, itemgetter
from toolz.utils import raises
from functools import partial
def test_curry_subclassable():

    class mycurry(curry):
        pass
    add = mycurry(lambda x, y: x + y)
    assert isinstance(add, curry)
    assert isinstance(add, mycurry)
    assert isinstance(add(1), mycurry)
    assert isinstance(add()(1), mycurry)
    assert add(1)(2) == 3
    '\n    class curry2(curry):\n        def _should_curry(self, args, kwargs, exc=None):\n            return len(self.args) + len(args) < 2\n\n    add = curry2(lambda x, y: x+y)\n    assert isinstance(add(1), curry2)\n    assert add(1)(2) == 3\n    assert isinstance(add(1)(x=2), curry2)\n    assert raises(TypeError, lambda: add(1)(x=2)(3))\n    '