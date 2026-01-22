import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def testAssertIs(self):
    self.assertMessages('assertIs', (None, 'foo'), ["^None is not 'foo'$", '^oops$', "^None is not 'foo'$", "^None is not 'foo' : oops$"])