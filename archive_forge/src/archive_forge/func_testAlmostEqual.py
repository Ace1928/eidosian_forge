import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product
def testAlmostEqual(self):
    self.assertMessages('assertAlmostEqual', (1, 2), ['^1 != 2 within 7 places \\(1 difference\\)$', '^oops$', '^1 != 2 within 7 places \\(1 difference\\)$', '^1 != 2 within 7 places \\(1 difference\\) : oops$'])