import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def testAssertWarns(self):
    self.assertMessagesCM('assertWarns', (UserWarning,), lambda: None, ['^UserWarning not triggered$', '^oops$', '^UserWarning not triggered$', '^UserWarning not triggered : oops$'])