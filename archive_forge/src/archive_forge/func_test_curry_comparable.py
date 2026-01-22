import inspect
import toolz
from toolz.functoolz import (thread_first, thread_last, memoize, curry,
from operator import add, mul, itemgetter
from toolz.utils import raises
from functools import partial
def test_curry_comparable():

    def foo(a, b, c=1):
        return a + b + c
    f1 = curry(foo, 1, c=2)
    f2 = curry(foo, 1, c=2)
    g1 = curry(foo, 1, c=3)
    h1 = curry(foo, c=2)
    h2 = h1(c=2)
    h3 = h1()
    assert f1 == f2
    assert not f1 != f2
    assert f1 != g1
    assert not f1 == g1
    assert f1 != h1
    assert h1 == h2
    assert h1 == h3

    def bar(a, b, c=1):
        return a + b + c
    b1 = curry(bar, 1, c=2)
    assert b1 != f1
    assert {f1, f2, g1, h1, h2, h3, b1, b1()} == {f1, g1, h1, b1}
    unhash1 = curry(foo, [])
    assert raises(TypeError, lambda: hash(unhash1))
    unhash2 = curry(foo, c=[])
    assert raises(TypeError, lambda: hash(unhash2))