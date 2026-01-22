import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.iis import write_iis
from pyomo.contrib.iis.iis import _supported_solvers
from pyomo.common.tempfiles import TempfileManager
import os
@unittest.skipIf(pyo.SolverFactory('cplex_persistent').available(exception_flag=False) or pyo.SolverFactory('gurobi_persistent').available(exception_flag=False) or pyo.SolverFactory('xpress_persistent').available(exception_flag=False), 'Persistent solver available')
def test_exception_iis_no_solver_available(self):
    with self.assertRaises(RuntimeError, msg=f'Could not find a solver to use, supported solvers are {_supported_solvers}'):
        _test_iis(None)