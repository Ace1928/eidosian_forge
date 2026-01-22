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
def test_without_initial_value(self):
    """Test default initialization"""
    self.model.x = Var(self.model.A, self.model.A)
    self.instance = self.model.create_instance()
    self.assertEqual(self.instance.x[1, 1].value, None)
    self.assertEqual(self.instance.x[2, 2].value, None)
    self.instance.x[1, 1] = 5
    self.instance.x[2, 2] = 6
    self.assertEqual(self.instance.x[1, 1].value, 5)
    self.assertEqual(self.instance.x[2, 2].value, 6)