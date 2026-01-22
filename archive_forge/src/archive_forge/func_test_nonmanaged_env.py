import gc
from unittest.mock import patch
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.environ import SolverFactory, ConcreteModel
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect
def test_nonmanaged_env(self):
    gp.setParam('IterationLimit', 0)
    gp.setParam('Presolve', 0)
    with SolverFactory('gurobi_direct') as opt:
        results = opt.solve(self.model)
        self.assertEqual(results.solver.status, SolverStatus.aborted)
        self.assertEqual(results.solver.termination_condition, TerminationCondition.maxIterations)