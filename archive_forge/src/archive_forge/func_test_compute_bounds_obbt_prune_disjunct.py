import pyomo.common.unittest as unittest
from pyomo.contrib.gdp_bounds.info import disjunctive_lb, disjunctive_ub
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import check_available_solvers
@unittest.skipIf('cbc' not in solvers, 'CBC solver not available')
def test_compute_bounds_obbt_prune_disjunct(self):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 1))
    m.d1 = Disjunct()
    m.d1.c = Constraint(expr=m.x >= 2)
    m.d1.c2 = Constraint(expr=m.x <= 1)
    m.d2 = Disjunct()
    m.d2.c = Constraint(expr=m.x + 3 == 0)
    m.disj = Disjunction(expr=[m.d1, m.d2])
    m.obj = Objective(expr=m.x)
    TransformationFactory('contrib.compute_disj_var_bounds').apply_to(m, solver='cbc')
    self.assertFalse(m.d1.active)
    self.assertEqual(m.d1.binary_indicator_var.value, 0)
    self.assertTrue(m.d1.indicator_var.fixed)