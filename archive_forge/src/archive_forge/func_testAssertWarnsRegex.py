import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def testAssertWarnsRegex(self):
    self.assertMessagesCM('assertWarnsRegex', (UserWarning, 'unused regex'), lambda: None, ['^UserWarning not triggered$', '^oops$', '^UserWarning not triggered$', '^UserWarning not triggered : oops$'])

    def raise_wrong_message():
        warnings.warn('foo')
    self.assertMessagesCM('assertWarnsRegex', (UserWarning, 'regex'), raise_wrong_message, ['^"regex" does not match "foo"$', '^oops$', '^"regex" does not match "foo"$', '^"regex" does not match "foo" : oops$'])