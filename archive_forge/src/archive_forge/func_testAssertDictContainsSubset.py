import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def testAssertDictContainsSubset(self):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        self.assertMessages('assertDictContainsSubset', ({'key': 'value'}, {}), ["^Missing: 'key'$", '^oops$', "^Missing: 'key'$", "^Missing: 'key' : oops$"])