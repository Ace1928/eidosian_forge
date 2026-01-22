from pyomo.common import unittest
import pyomo.environ as pyo
from pyomo.contrib import appsi
from pyomo.contrib.appsi.cmodel import cmodel_available
from pyomo.contrib.fbbt.tests.test_fbbt import FbbtTestBase
from pyomo.common.errors import InfeasibleConstraintException
import math
def test_sync_after_infeasible(self):
    m = pe.ConcreteModel()
    m.x = pe.Var(bounds=(1, 1))
    m.y = pe.Var()
    m.c1 = pe.Constraint(expr=m.x == m.y)
    m.c2 = pe.Constraint(expr=m.y == 2)
    it = appsi.fbbt.IntervalTightener()
    try:
        it.perform_fbbt(m)
        was_infeasible = False
    except InfeasibleConstraintException:
        was_infeasible = True
    self.assertTrue(was_infeasible)
    self.assertAlmostEqual(m.x.lb, 1)
    self.assertAlmostEqual(m.x.ub, 1)
    self.assertAlmostEqual(m.y.lb, 1)
    self.assertAlmostEqual(m.y.ub, 1)
    m = pe.ConcreteModel()
    m.x = pe.Var(bounds=(1, 1))
    m.y = pe.Var()
    m.c1 = pe.Constraint(expr=m.x == m.y)
    m.c2 = pe.Constraint(expr=m.y == 2)
    it = appsi.fbbt.IntervalTightener()
    try:
        it.perform_fbbt_with_seed(m, m.x)
        was_infeasible = False
    except InfeasibleConstraintException:
        was_infeasible = True
    self.assertTrue(was_infeasible)
    self.assertAlmostEqual(m.x.lb, 1)
    self.assertAlmostEqual(m.x.ub, 1)
    self.assertAlmostEqual(m.y.lb, 1)
    self.assertAlmostEqual(m.y.ub, 1)