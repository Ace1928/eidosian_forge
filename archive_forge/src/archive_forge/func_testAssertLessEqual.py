import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def testAssertLessEqual(self):
    self.assertMessages('assertLessEqual', (2, 1), ['^2 not less than or equal to 1$', '^oops$', '^2 not less than or equal to 1$', '^2 not less than or equal to 1 : oops$'])