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
def test_eq1(self):
    """Various checks for set equality and inequality (1)"""
    self.assertTrue(not self.instance.A == self.instance.tmpset1)
    self.assertTrue(not self.instance.tmpset1 == self.instance.A)
    self.assertTrue(self.instance.A != self.instance.tmpset1)
    self.assertTrue(self.instance.tmpset1 != self.instance.A)