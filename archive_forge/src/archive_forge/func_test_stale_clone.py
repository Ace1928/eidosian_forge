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
def test_stale_clone(self):
    m = ConcreteModel()
    m.x = Var(initialize=0)
    self.assertFalse(m.x.stale)
    m.y = Var()
    self.assertTrue(m.y.stale)
    m.z = Var(initialize=0)
    self.assertFalse(m.z.stale)
    i = m.clone()
    self.assertFalse(i.x.stale)
    self.assertTrue(i.y.stale)
    self.assertFalse(i.z.stale)
    StaleFlagManager.mark_all_as_stale(delayed=True)
    m.z = 5
    i = m.clone()
    self.assertTrue(i.x.stale)
    self.assertTrue(i.y.stale)
    self.assertFalse(i.z.stale)