import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def test_assertRaises_frames_survival(self):

    class A:
        pass
    wr = None

    class Foo(unittest.TestCase):

        def foo(self):
            nonlocal wr
            a = A()
            wr = weakref.ref(a)
            try:
                raise OSError
            except OSError:
                raise ValueError

        def test_functional(self):
            self.assertRaises(ValueError, self.foo)

        def test_with(self):
            with self.assertRaises(ValueError):
                self.foo()
    Foo('test_functional').run()
    gc_collect()
    self.assertIsNone(wr())
    Foo('test_with').run()
    gc_collect()
    self.assertIsNone(wr())