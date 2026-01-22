import gc
from unittest.mock import patch
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.environ import SolverFactory, ConcreteModel
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect
def test_set_environment_parameters(self):
    with SolverFactory('gurobi_direct', manage_env=True, options={'ComputeServer': 'my-cs-url'}) as opt:
        with self.assertRaisesRegex(ApplicationError, 'my-cs-url'):
            opt.solve(self.model)