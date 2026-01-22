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
def test_prodset(self):
    a = Set(initialize=[1, 2])
    a.construct()
    b = Set(initialize=[6, 7])
    b.construct()
    c = a * b
    c.construct()
    self.assertEqual((6, 2) in c, False)
    c = pyomo.core.base.set.SetProduct(a, b)
    c.virtual = True
    self.assertEqual((6, 2) in c, False)
    self.assertEqual((1, 7) in c, True)