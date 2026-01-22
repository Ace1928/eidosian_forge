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
@unittest.skipIf(not check_available_solvers('ipopt'), "The 'ipopt' solver is not available")
def test_clone_gsl_function(self):
    DLL = find_GSL()
    if not DLL:
        self.skipTest('Could not find the amplgsl.dll library')
    m = ConcreteModel()
    m.z_func = ExternalFunction(library=DLL, function='gsl_sf_gamma')
    self.assertIsInstance(m.z_func, AMPLExternalFunction)
    m.x = Var(initialize=3, bounds=(1e-05, None))
    m.o = Objective(expr=m.z_func(m.x))
    opt = SolverFactory('ipopt')
    model2 = m.clone()
    res = opt.solve(model2, tee=True)
    self.assertAlmostEqual(value(model2.o), 0.885603194411, 7)
    self.assertAlmostEqual(value(m.o), 2)
    model3 = m.clone()
    res = opt.solve(model3, tee=True)
    self.assertAlmostEqual(value(model3.o), 0.885603194411, 7)