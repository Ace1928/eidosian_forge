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
def test_MutableParams(self):
    model = ConcreteModel()
    model.lb = Param(initialize=1, mutable=True)
    model.ub = Param(initialize=5, mutable=True)
    model.A = RangeSet(model.lb, model.ub)
    self.assertEqual(set(model.A.data()), set([1, 2, 3, 4, 5]))
    model.lb = 2
    model.ub = 4
    model.B = RangeSet(model.lb, model.ub)
    self.assertEqual(set(model.A.data()), set([1, 2, 3, 4, 5]))
    self.assertEqual(set(model.B.data()), set([2, 3, 4]))