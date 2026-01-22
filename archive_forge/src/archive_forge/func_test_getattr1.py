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
def test_getattr1(self):
    """
        Verify the behavior of non-standard suffixes with simple variable
        """
    model = AbstractModel()
    model.a = Var()
    model.suffix = Suffix(datatype=Suffix.INT)
    instance = model.create_instance()
    self.assertEqual(instance.suffix.get(instance.a), None)
    instance.suffix.set_value(instance.a, True)
    self.assertEqual(instance.suffix.get(instance.a), True)