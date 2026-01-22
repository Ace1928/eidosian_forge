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
def test_upper_bound(self):
    m = ConcreteModel()
    m.x = Var()
    m.p = Param(mutable=True, initialize=2)
    self.assertIsNone(m.x.upper)
    m.x.domain = NonPositiveReals
    self.assertIs(type(m.x.upper), int)
    self.assertEqual(value(m.x.upper), 0)
    m.x.domain = Reals
    m.x.setub(-5 * m.p)
    self.assertIs(type(m.x.upper), NPV_ProductExpression)
    self.assertEqual(value(m.x.upper), -10)
    m.x.domain = NonPositiveReals
    self.assertIs(type(m.x.upper), NPV_MinExpression)
    self.assertEqual(value(m.x.upper), -10)
    with self.assertRaisesRegex(ValueError, "Potentially variable input of type 'ScalarVar' supplied as upper bound for variable 'x'"):
        m.x.setub(m.x)