import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.contrib.solver.gurobi import Gurobi
from pyomo.contrib.solver.results import SolutionStatus
from pyomo.core.expr.taylor_series import taylor_series_expansion
import gurobipy
def test_zero_time_limit(self):
    m = create_pmedian_model()
    opt = Gurobi()
    opt.config.time_limit = 0
    opt.config.load_solutions = False
    opt.config.raise_exception_on_nonoptimal_result = False
    res = opt.solve(m)
    num_solutions = opt.get_model_attr('SolCount')
    if num_solutions == 0:
        self.assertIsNone(res.incumbent_objective)