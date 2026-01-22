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
def test_mutable_self4(self):
    model = ConcreteModel()
    model.P = Param([1, 2], default=1.0, mutable=True)
    self.assertEqual(model.P[1].value, 1.0)
    self.assertEqual(model.P[2].value, 1.0)
    model.P[1].value = 0.0
    self.assertEqual(model.P[1].value, 0.0)
    self.assertEqual(model.P[2].value, 1.0)
    model.Q = Param([1, 2], default=1.0, mutable=True)
    self.assertEqual(model.Q[1].value, 1.0)
    self.assertEqual(model.Q[2].value, 1.0)
    model.Q[1] = 0.0
    self.assertEqual(model.Q[1].value, 0.0)
    self.assertEqual(model.Q[2].value, 1.0)