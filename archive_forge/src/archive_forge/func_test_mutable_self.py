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
def test_mutable_self(self):
    model = ConcreteModel()
    model.Q = Param(initialize=0.0, mutable=True)
    model.x = Var()
    model.CON = Constraint(expr=model.Q <= model.x)
    self.assertEqual(0.0, value(model.CON[None].lower))
    model.Q = 1.0
    self.assertEqual(1.0, value(model.CON[None].lower))