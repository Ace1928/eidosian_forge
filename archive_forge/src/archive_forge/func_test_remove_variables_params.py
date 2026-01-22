import pyomo.environ as pe
import pyomo.common.unittest as unittest
from pyomo.contrib.appsi.base import TerminationCondition, Results, PersistentSolver
from pyomo.contrib.appsi.solvers.wntr import Wntr, wntr_available
import math
def test_remove_variables_params(self):
    m = pe.ConcreteModel()
    m.x = pe.Var()
    m.y = pe.Var()
    m.z = pe.Var()
    m.z.fix(0)
    m.px = pe.Param(mutable=True, initialize=1)
    m.py = pe.Param(mutable=True, initialize=1)
    m.c1 = pe.Constraint(expr=m.x == m.px)
    m.c2 = pe.Constraint(expr=m.y == m.py)
    opt = Wntr()
    opt.wntr_options.update(_default_wntr_options)
    res = opt.solve(m)
    self.assertEqual(res.termination_condition, TerminationCondition.optimal)
    self.assertAlmostEqual(m.x.value, 1)
    self.assertAlmostEqual(m.y.value, 1)
    self.assertAlmostEqual(m.z.value, 0)
    del m.c2
    del m.y
    del m.py
    m.z.value = 2
    m.px.value = 2
    res = opt.solve(m)
    self.assertEqual(res.termination_condition, TerminationCondition.optimal)
    self.assertAlmostEqual(m.x.value, 2)
    self.assertAlmostEqual(m.z.value, 2)
    del m.z
    m.px.value = 3
    res = opt.solve(m)
    self.assertEqual(res.termination_condition, TerminationCondition.optimal)
    self.assertAlmostEqual(m.x.value, 3)