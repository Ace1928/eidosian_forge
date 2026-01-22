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
def test_0based_initialize_with_dict(self):
    """Test initialize option with a dictionary"""
    self.model.x = VarList(initialize={1: 1.3, 2: 2.3}, starting_index=0)
    self.assertRaisesRegex(KeyError, ".*Index '2' is not valid for indexed component 'x'", self.model.create_instance)