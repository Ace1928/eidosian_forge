import os
import shutil
import sys
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.gsl import find_GSL
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import (
from pyomo.core.base.external import (
from pyomo.core.base.units_container import pint_available, units
from pyomo.core.expr.numeric_expr import (
from pyomo.opt import check_available_solvers
def test_call_countArgs(self):
    m = ConcreteModel()
    m.f = ExternalFunction(_count)
    self.assertIsInstance(m.f, PythonCallbackFunction)
    self.assertEqual(value(m.f()), 0)
    self.assertEqual(value(m.f(2)), 1)
    self.assertEqual(value(m.f(2, 3)), 2)