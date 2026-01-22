import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.contrib.solver.gurobi import Gurobi
from pyomo.contrib.solver.results import SolutionStatus
from pyomo.core.expr.taylor_series import taylor_series_expansion
import gurobipy
def test_nonconvex_qcp_objective_bound_2(self):
    m = pe.ConcreteModel()
    m.x = pe.Var(bounds=(-5, 5))
    m.y = pe.Var(bounds=(-5, 5))
    m.obj = pe.Objective(expr=-m.x ** 2 - m.y)
    m.c1 = pe.Constraint(expr=m.y <= -2 * m.x + 1)
    m.c2 = pe.Constraint(expr=m.y <= m.x - 2)
    opt = Gurobi()
    opt.config.solver_options['nonconvex'] = 2
    opt.config.solver_options['MIPGap'] = 0.5
    res = opt.solve(m)
    self.assertAlmostEqual(res.incumbent_objective, -4)
    self.assertAlmostEqual(res.objective_bound, -6)