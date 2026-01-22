import pyomo.environ as pe
import pyomo.common.unittest as unittest
from pyomo.contrib.appsi.base import TerminationCondition, Results, PersistentSolver
from pyomo.contrib.appsi.solvers.wntr import Wntr, wntr_available
import math
def test_get_primals(self):
    m = pe.ConcreteModel()
    m.x = pe.Var()
    m.y = pe.Var()
    m.c1 = pe.Constraint(expr=m.y == (m.x - 1) ** 2)
    m.c2 = pe.Constraint(expr=m.y == pe.exp(m.x))
    opt = Wntr()
    opt.config.load_solution = False
    opt.wntr_options.update(_default_wntr_options)
    res = opt.solve(m)
    self.assertEqual(res.termination_condition, TerminationCondition.optimal)
    self.assertAlmostEqual(m.x.value, None)
    self.assertAlmostEqual(m.y.value, None)
    primals = opt.get_primals()
    self.assertAlmostEqual(primals[m.x], 0)
    self.assertAlmostEqual(primals[m.y], 1)