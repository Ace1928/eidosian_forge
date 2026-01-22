import inspect
import toolz
from toolz.functoolz import (thread_first, thread_last, memoize, curry,
from operator import add, mul, itemgetter
from toolz.utils import raises
from functools import partial
def test_memoize_on_classmethods():

    class A(object):
        BASE = 10
        HASH = 10

        def __init__(self, base):
            self.BASE = base

        @memoize
        def addmethod(self, x, y):
            return self.BASE + x + y

        @classmethod
        @memoize
        def addclass(cls, x, y):
            return cls.BASE + x + y

        @staticmethod
        @memoize
        def addstatic(x, y):
            return x + y

        def __hash__(self):
            return self.HASH
    a = A(100)
    assert a.addmethod(3, 4) == 107
    assert A.addmethod(a, 3, 4) == 107
    a.BASE = 200
    assert a.addmethod(3, 4) == 107
    a.HASH = 200
    assert a.addmethod(3, 4) == 207
    assert a.addclass(3, 4) == 17
    assert A.addclass(3, 4) == 17
    A.BASE = 20
    assert A.addclass(3, 4) == 17
    A.HASH = 20
    assert A.addclass(3, 4) == 17
    assert a.addstatic(3, 4) == 7
    assert A.addstatic(3, 4) == 7