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
def test_lower_bound(self):
    m = ConcreteModel()
    m.x = Var()
    m.p = Param(mutable=True, initialize=2)
    self.assertIsNone(m.x.lower)
    m.x.domain = NonNegativeReals
    self.assertIs(type(m.x.lower), int)
    self.assertEqual(value(m.x.lower), 0)
    m.x.domain = Reals
    m.x.setlb(5 * m.p)
    self.assertIs(type(m.x.lower), NPV_ProductExpression)
    self.assertEqual(value(m.x.lower), 10)
    m.x.domain = NonNegativeReals
    self.assertIs(type(m.x.lower), NPV_MaxExpression)
    self.assertEqual(value(m.x.lower), 10)
    with self.assertRaisesRegex(ValueError, "Potentially variable input of type 'ScalarVar' supplied as lower bound for variable 'x'"):
        m.x.setlb(m.x)