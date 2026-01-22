import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core.base import IntegerSet
from pyomo.core.expr.numeric_expr import (
from pyomo.core.staleflag import StaleFlagManager
from pyomo.environ import (
from pyomo.core.base.units_container import units, pint_available, UnitsError
def test_upper_bound_setter(self):
    m = ConcreteModel()
    m.x = Var()
    self.assertIsNone(m.x.ub)
    m.x.ub = 1
    self.assertEqual(m.x.ub, 1)
    m.x.upper = 2
    self.assertEqual(m.x.ub, 2)
    m.x.setub(3)
    self.assertEqual(m.x.ub, 3)
    m.y = Var([1])
    self.assertIsNone(m.y[1].ub)
    m.y[1].ub = 1
    self.assertEqual(m.y[1].ub, 1)
    m.y[1].upper = 2
    self.assertEqual(m.y[1].ub, 2)
    m.y[1].setub(3)
    self.assertEqual(m.y[1].ub, 3)