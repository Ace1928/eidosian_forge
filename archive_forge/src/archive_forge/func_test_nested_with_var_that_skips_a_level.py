from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.environ import (
from pyomo.core.expr.compare import (
import pyomo.core.expr as EXPR
from pyomo.core.base import constraint
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import random
from io import StringIO
import os
from os.path import abspath, dirname, join
from filecmp import cmp
def test_nested_with_var_that_skips_a_level(self):
    m = ConcreteModel()
    m.x = Var(bounds=(-2, 9))
    m.y = Var(bounds=(-3, 8))
    m.y1 = Disjunct()
    m.y1.c1 = Constraint(expr=m.x >= 4)
    m.y1.z1 = Disjunct()
    m.y1.z1.c1 = Constraint(expr=m.y == 2)
    m.y1.z1.w1 = Disjunct()
    m.y1.z1.w1.c1 = Constraint(expr=m.x == 3)
    m.y1.z1.w2 = Disjunct()
    m.y1.z1.w2.c1 = Constraint(expr=m.x >= 1)
    m.y1.z1.disjunction = Disjunction(expr=[m.y1.z1.w1, m.y1.z1.w2])
    m.y1.z2 = Disjunct()
    m.y1.z2.c1 = Constraint(expr=m.y == 1)
    m.y1.disjunction = Disjunction(expr=[m.y1.z1, m.y1.z2])
    m.y2 = Disjunct()
    m.y2.c1 = Constraint(expr=m.x == 4)
    m.disjunction = Disjunction(expr=[m.y1, m.y2])
    hull = TransformationFactory('gdp.hull')
    hull.apply_to(m)
    x_y1 = hull.get_disaggregated_var(m.x, m.y1)
    x_y2 = hull.get_disaggregated_var(m.x, m.y2)
    x_z1 = hull.get_disaggregated_var(m.x, m.y1.z1)
    x_z2 = hull.get_disaggregated_var(m.x, m.y1.z2)
    x_w1 = hull.get_disaggregated_var(m.x, m.y1.z1.w1)
    x_w2 = hull.get_disaggregated_var(m.x, m.y1.z1.w2)
    y_z1 = hull.get_disaggregated_var(m.y, m.y1.z1)
    y_z2 = hull.get_disaggregated_var(m.y, m.y1.z2)
    y_y1 = hull.get_disaggregated_var(m.y, m.y1)
    y_y2 = hull.get_disaggregated_var(m.y, m.y2)
    cons = hull.get_disaggregation_constraint(m.x, m.y1.z1.disjunction)
    self.assertTrue(cons.active)
    cons_expr = self.simplify_cons(cons)
    assertExpressionsEqual(self, cons_expr, x_z1 - x_w1 - x_w2 == 0.0)
    cons = hull.get_disaggregation_constraint(m.x, m.y1.disjunction)
    self.assertTrue(cons.active)
    cons_expr = self.simplify_cons(cons)
    assertExpressionsEqual(self, cons_expr, x_y1 - x_z2 - x_z1 == 0.0)
    cons = hull.get_disaggregation_constraint(m.x, m.disjunction)
    self.assertTrue(cons.active)
    cons_expr = self.simplify_cons(cons)
    assertExpressionsEqual(self, cons_expr, m.x - x_y1 - x_y2 == 0.0)
    cons = hull.get_disaggregation_constraint(m.y, m.y1.z1.disjunction, raise_exception=False)
    self.assertIsNone(cons)
    cons = hull.get_disaggregation_constraint(m.y, m.y1.disjunction)
    self.assertTrue(cons.active)
    cons_expr = self.simplify_cons(cons)
    assertExpressionsEqual(self, cons_expr, y_y1 - y_z1 - y_z2 == 0.0)
    cons = hull.get_disaggregation_constraint(m.y, m.disjunction)
    self.assertTrue(cons.active)
    cons_expr = self.simplify_cons(cons)
    assertExpressionsEqual(self, cons_expr, m.y - y_y2 - y_y1 == 0.0)