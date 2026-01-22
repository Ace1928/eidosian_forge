import inspect
import toolz
from toolz.functoolz import (thread_first, thread_last, memoize, curry,
from operator import add, mul, itemgetter
from toolz.utils import raises
from functools import partial
def test_curry_on_classmethods():

    class A(object):
        BASE = 10

        def __init__(self, base):
            self.BASE = base

        @curry
        def addmethod(self, x, y):
            return self.BASE + x + y

        @classmethod
        @curry
        def addclass(cls, x, y):
            return cls.BASE + x + y

        @staticmethod
        @curry
        def addstatic(x, y):
            return x + y
    a = A(100)
    assert a.addmethod(3, 4) == 107
    assert a.addmethod(3)(4) == 107
    assert A.addmethod(a, 3, 4) == 107
    assert A.addmethod(a)(3)(4) == 107
    assert a.addclass(3, 4) == 17
    assert a.addclass(3)(4) == 17
    assert A.addclass(3, 4) == 17
    assert A.addclass(3)(4) == 17
    assert a.addstatic(3, 4) == 7
    assert a.addstatic(3)(4) == 7
    assert A.addstatic(3, 4) == 7
    assert A.addstatic(3)(4) == 7
    assert isinstance(a.addmethod, curry)
    assert isinstance(A.addmethod, curry)