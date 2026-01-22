import math
import os
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import PyomoException
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.param import _ParamData
from pyomo.core.base.set import _SetData
from pyomo.core.base.units_container import units, pint_available, UnitsError
from io import StringIO
def test_mutable_self2(self):
    model = ConcreteModel()
    model.P = Param([1], initialize=1.0, mutable=True)
    model.x = Var()
    model.CON = Constraint(expr=model.P[1] <= model.x)
    self.assertEqual(1.0, value(model.CON[None].lower))
    model.P[1] = 2.0
    self.assertEqual(2.0, value(model.CON[None].lower))