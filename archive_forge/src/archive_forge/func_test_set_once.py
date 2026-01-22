import gc
from unittest.mock import patch
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.environ import SolverFactory, ConcreteModel
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect
def test_set_once(self):
    envparams = {}
    modelparams = {}

    class TempEnv(gp.Env):

        def setParam(self, param, value):
            envparams[param] = value

    class TempModel(gp.Model):

        def setParam(self, param, value):
            modelparams[param] = value
    with patch('gurobipy.Env', new=TempEnv), patch('gurobipy.Model', new=TempModel):
        with SolverFactory('gurobi_direct', options={'Method': 2, 'MIPFocus': 1}, manage_env=True) as opt:
            opt.solve(self.model, options={'MIPFocus': 2})
    assert envparams == {'Method': 2, 'MIPFocus': 1}
    assert modelparams == {'MIPFocus': 2, 'OutputFlag': 0}