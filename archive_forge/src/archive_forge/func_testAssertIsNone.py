import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def testAssertIsNone(self):
    self.assertMessages('assertIsNone', ('not None',), ["^'not None' is not None$", '^oops$', "^'not None' is not None$", "^'not None' is not None : oops$"])