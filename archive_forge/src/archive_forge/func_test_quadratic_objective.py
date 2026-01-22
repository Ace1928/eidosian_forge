import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.contrib.solver.gurobi import Gurobi
from pyomo.contrib.solver.results import SolutionStatus
from pyomo.core.expr.taylor_series import taylor_series_expansion
import gurobipy
def test_quadratic_objective(self):
    m = pe.ConcreteModel()
    m.a = pe.Param(initialize=1, mutable=True)
    m.b = pe.Param(initialize=1, mutable=True)
    m.c = pe.Param(initialize=1, mutable=True)
    m.x = pe.Var()
    m.obj = pe.Objective(expr=m.a * m.x ** 2 + m.b * m.x + m.c)
    opt = Gurobi()
    res = opt.solve(m)
    self.assertAlmostEqual(m.x.value, -m.b.value / (2 * m.a.value))
    self.assertAlmostEqual(res.incumbent_objective, m.a.value * m.x.value ** 2 + m.b.value * m.x.value + m.c.value)
    m.a.value = 2
    m.b.value = 4
    m.c.value = -1
    res = opt.solve(m)
    self.assertAlmostEqual(m.x.value, -m.b.value / (2 * m.a.value))
    self.assertAlmostEqual(res.incumbent_objective, m.a.value * m.x.value ** 2 + m.b.value * m.x.value + m.c.value)