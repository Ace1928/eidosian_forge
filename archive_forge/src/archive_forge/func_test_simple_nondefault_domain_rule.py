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
def test_simple_nondefault_domain_rule(self):
    model = ConcreteModel()
    model.x = Var(domain=lambda m: Integers)
    self.assertIs(model.x.domain, Integers)