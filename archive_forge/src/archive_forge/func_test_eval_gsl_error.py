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
def test_eval_gsl_error(self):
    DLL = find_GSL()
    if not DLL:
        self.skipTest('Could not find the amplgsl.dll library')
    model = ConcreteModel()
    model.bogus = ExternalFunction(library=DLL, function='bogus_function')
    with self.assertRaisesRegex(RuntimeError, "Error: external function 'bogus_function' was not registered within external library(?s:.*)gsl_sf_gamma"):
        f = model.bogus.evaluate((1,))