import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def testAssertIsNotNone(self):
    self.assertMessages('assertIsNotNone', (None,), ['^unexpectedly None$', '^oops$', '^unexpectedly None$', '^unexpectedly None : oops$'])