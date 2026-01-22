import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.contrib.solver.gurobi import Gurobi
from pyomo.contrib.solver.results import SolutionStatus
from pyomo.core.expr.taylor_series import taylor_series_expansion
import gurobipy
def test_fixed_var(self):
    m = pe.ConcreteModel()
    m.a = pe.Param(initialize=1, mutable=True)
    m.b = pe.Param(initialize=1, mutable=True)
    m.c = pe.Param(initialize=1, mutable=True)
    m.x = pe.Var()
    m.y = pe.Var()
    m.obj = pe.Objective(expr=m.y)
    m.con = pe.Constraint(expr=m.y >= m.a * m.x ** 2 + m.b * m.x + m.c)
    m.x.fix(1)
    opt = Gurobi()
    res = opt.solve(m)
    self.assertAlmostEqual(m.x.value, 1)
    self.assertAlmostEqual(m.y.value, 3)
    m.x.value = 2
    res = opt.solve(m)
    self.assertAlmostEqual(m.x.value, 2)
    self.assertAlmostEqual(m.y.value, 7)
    m.x.unfix()
    res = opt.solve(m)
    self.assertAlmostEqual(m.x.value, -m.b.value / (2 * m.a.value))
    self.assertAlmostEqual(m.y.value, m.a.value * m.x.value ** 2 + m.b.value * m.x.value + m.c.value)