from io import StringIO
import sys
import logging
import pyomo.common.unittest as unittest
from pyomo.contrib.trustregion.util import IterationLogger, minIgnoreNone, maxIgnoreNone
from pyomo.common.log import LoggingIntercept
def test_minIgnoreNone(self):
    a = 1
    b = 2
    self.assertEqual(minIgnoreNone(a, b), a)
    a = None
    self.assertEqual(minIgnoreNone(a, b), b)
    a = 1
    b = None
    self.assertEqual(minIgnoreNone(a, b), a)
    a = None
    self.assertEqual(minIgnoreNone(a, b), None)