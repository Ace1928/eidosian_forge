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
def test_ub_attr2(self):
    """Test ub attribute"""
    self.model.x = Var(within=NonPositiveReals, bounds=(-2, 1))
    self.instance = self.model.create_instance()
    self.assertEqual(value(self.instance.x.lb), -2.0)
    self.assertEqual(value(self.instance.x.ub), 0.0)