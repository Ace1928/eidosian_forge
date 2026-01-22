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
def test_Expressions(self):
    model = ConcreteModel()
    model.p = Param(initialize=1, mutable=True)
    model.lb = Expression(expr=model.p * 2 - 1)
    model.ub = Expression(expr=model.p * 5)
    model.A = RangeSet(model.lb, model.ub)
    self.assertEqual(set(model.A.data()), set([1, 2, 3, 4, 5]))
    model.p = 2
    model.B = RangeSet(model.lb, model.ub)
    self.assertEqual(set(model.A.data()), set([1, 2, 3, 4, 5]))
    self.assertEqual(set(model.B.data()), set([3, 4, 5, 6, 7, 8, 9, 10]))