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
def test_NegativeIntegers(self):
    x = NegativeIntegers
    self.assertFalse(None in x)
    self.assertFalse(10 in x)
    self.assertFalse(1.1 in x)
    self.assertFalse(1 in x)
    self.assertFalse(0.3 in x)
    self.assertFalse(0 in x)
    self.assertFalse(-0.45 in x)
    self.assertTrue(-1 in x)
    self.assertFalse(-2.2 in x)
    self.assertTrue(-10 in x)