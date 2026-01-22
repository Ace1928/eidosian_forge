import math
import pickle
from pyomo.common.errors import PyomoException
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.log import LoggingIntercept
from pyomo.util.check_units import assert_units_consistent, check_units_equivalent
from pyomo.core.expr import inequality
from pyomo.core.expr.numvalue import NumericConstant
import pyomo.core.expr as EXPR
from pyomo.core.base.units_container import (
from io import StringIO
def test_dimensionless(self):
    uc = units
    kg = uc.kg
    dless = uc.dimensionless
    self._get_check_units_ok(2.0 == 2.0 * dless, uc, 'dimensionless', EXPR.EqualityExpression)
    x = uc.get_units(2.0)
    self.assertIs(type(x), _PyomoUnit)
    self.assertEqual(x, dless)
    x = uc.get_units(2.0 * dless)
    self.assertIs(type(x), _PyomoUnit)
    self.assertEqual(x, dless)
    x = uc.get_units(kg / kg)
    self.assertIs(type(x), _PyomoUnit)
    self.assertEqual(x, dless)