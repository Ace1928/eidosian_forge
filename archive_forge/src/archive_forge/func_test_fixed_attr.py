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
def test_fixed_attr(self):
    """Test fixed attribute"""
    self.model.x = Var(self.model.A, self.model.A)
    self.model.y = Var(self.model.A, self.model.A)
    self.instance = self.model.create_instance()
    self.instance.x.fixed = True
    self.assertEqual(self.instance.x[1, 2].fixed, False)
    self.instance.y[1, 2].fixed = True
    self.assertEqual(self.instance.y[1, 2].fixed, True)