import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.core.expr.taylor_series import taylor_series_expansion
@unittest.skipIf(not gurobipy_available, 'gurobipy is not available')
def test_quadratic_constraint_attr(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.c = pyo.Constraint(expr=m.y >= m.x ** 2)
    opt = pyo.SolverFactory('gurobi_persistent')
    opt.set_instance(m)
    self.assertEqual(opt.get_quadratic_constraint_attr(m.c, 'QCRHS'), 0)