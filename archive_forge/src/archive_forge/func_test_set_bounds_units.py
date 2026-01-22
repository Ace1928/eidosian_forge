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
@unittest.skipUnless(pint_available, 'units test requires pint module')
def test_set_bounds_units(self):
    m = ConcreteModel()
    m.x = Var(units=units.g)
    m.p = Param(mutable=True, initialize=1, units=units.kg)
    m.x.setlb(5)
    self.assertEqual(m.x.lb, 5)
    m.x.setlb(6 * units.g)
    self.assertEqual(m.x.lb, 6)
    m.x.setlb(7 * units.kg)
    self.assertEqual(m.x.lb, 7000)
    with self.assertRaises(UnitsError):
        m.x.setlb(1 * units.s)
    m.x.setlb(m.p)
    self.assertEqual(m.x.lb, 1000)
    m.p = 2 * units.kg
    self.assertEqual(m.x.lb, 2000)
    m.x.setub(2)
    self.assertEqual(m.x.ub, 2)
    m.x.setub(3 * units.g)
    self.assertEqual(m.x.ub, 3)
    m.x.setub(4 * units.kg)
    self.assertEqual(m.x.ub, 4000)
    with self.assertRaises(UnitsError):
        m.x.setub(1 * units.s)
    m.x.setub(m.p)
    self.assertEqual(m.x.ub, 2000)
    m.p = 3 * units.kg
    self.assertEqual(m.x.ub, 3000)