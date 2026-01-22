import copy
import itertools
import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.core.base
from pyomo.core.base.util import flatten_tuple
from pyomo.environ import (
from pyomo.core.base.set import _AnySet, RangeDifferenceError
def test_IntegerInterval(self):
    x = IntegerInterval()
    self.assertFalse(None in x)
    self.assertEqual(x.name, "'IntegerInterval(None, None)'")
    self.assertEqual(x.local_name, 'IntegerInterval(None, None)')
    self.assertTrue(10 in x)
    self.assertFalse(1.1 in x)
    self.assertTrue(1 in x)
    self.assertFalse(0.3 in x)
    self.assertTrue(0 in x)
    self.assertFalse(-0.45 in x)
    self.assertTrue(-1 in x)
    self.assertFalse(-2.2 in x)
    self.assertTrue(-10 in x)
    x = IntegerInterval(bounds=(-1, 1))
    self.assertFalse(None in x)
    self.assertEqual(x.name, "'IntegerInterval(-1, 1)'")
    self.assertEqual(x.local_name, 'IntegerInterval(-1, 1)')
    self.assertFalse(10 in x)
    self.assertFalse(1.1 in x)
    self.assertTrue(1 in x)
    self.assertFalse(0.3 in x)
    self.assertTrue(0 in x)
    self.assertFalse(-0.45 in x)
    self.assertTrue(-1 in x)
    self.assertFalse(-2.2 in x)
    self.assertFalse(-10 in x)
    x = IntegerInterval(bounds=(-1, 1), name='JUNK')
    self.assertFalse(None in x)
    self.assertEqual(x.name, 'JUNK')
    self.assertFalse(10 in x)
    self.assertFalse(1.1 in x)
    self.assertTrue(1 in x)
    self.assertFalse(0.3 in x)
    self.assertTrue(0 in x)
    self.assertFalse(-0.45 in x)
    self.assertTrue(-1 in x)
    self.assertFalse(-2.2 in x)
    self.assertFalse(-10 in x)