import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.iis import write_iis
from pyomo.contrib.iis.iis import _supported_solvers
from pyomo.common.tempfiles import TempfileManager
import os
@unittest.skipUnless(pyo.SolverFactory('cplex_persistent').available(exception_flag=False), 'CPLEX not available')
def test_write_iis_cplex(self):
    _test_iis('cplex')