import pyomo.environ as pe
import pyomo.common.unittest as unittest
from pyomo.contrib.appsi.base import TerminationCondition, Results, PersistentSolver
from pyomo.contrib.appsi.solvers.wntr import Wntr, wntr_available
import math
def test_remove_add_constraint(self):
    m = pe.ConcreteModel()
    m.x = pe.Var()
    m.y = pe.Var()
    m.c1 = pe.Constraint(expr=m.y == (m.x - 1) ** 2)
    m.c2 = pe.Constraint(expr=m.y == pe.exp(m.x))
    opt = Wntr()
    opt.config.symbolic_solver_labels = True
    opt.wntr_options.update(_default_wntr_options)
    res = opt.solve(m)
    self.assertEqual(res.termination_condition, TerminationCondition.optimal)
    self.assertAlmostEqual(m.x.value, 0)
    self.assertAlmostEqual(m.y.value, 1)
    del m.c2
    m.c2 = pe.Constraint(expr=m.y == pe.log(m.x))
    m.x.value = 0.5
    m.y.value = 0.5
    res = opt.solve(m)
    self.assertEqual(res.termination_condition, TerminationCondition.optimal)
    self.assertAlmostEqual(m.x.value, 1)
    self.assertAlmostEqual(m.y.value, 0)