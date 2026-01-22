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
def test_nested_with_var_that_does_not_appear_in_every_disjunct(self):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 10))
    m.y = Var(bounds=(-4, 5))
    m.parent1 = Disjunct()
    m.parent2 = Disjunct()
    m.parent2.c = Constraint(expr=m.x == 0)
    m.parent_disjunction = Disjunction(expr=[m.parent1, m.parent2])
    m.child1 = Disjunct()
    m.child1.c = Constraint(expr=m.x <= 8)
    m.child2 = Disjunct()
    m.child2.c = Constraint(expr=m.x + m.y <= 3)
    m.child3 = Disjunct()
    m.child3.c = Constraint(expr=m.x <= 7)
    m.parent1.disjunction = Disjunction(expr=[m.child1, m.child2, m.child3])
    hull = TransformationFactory('gdp.hull')
    hull.apply_to(m)
    y_c2 = hull.get_disaggregated_var(m.y, m.child2)
    self.assertEqual(y_c2.bounds, (-4, 5))
    other_y = hull.get_disaggregated_var(m.y, m.child1)
    self.assertEqual(other_y.bounds, (-4, 5))
    other_other_y = hull.get_disaggregated_var(m.y, m.child3)
    self.assertIs(other_y, other_other_y)
    y_p1 = hull.get_disaggregated_var(m.y, m.parent1)
    self.assertEqual(y_p1.bounds, (-4, 5))
    y_p2 = hull.get_disaggregated_var(m.y, m.parent2)
    self.assertEqual(y_p2.bounds, (-4, 5))
    y_cons = hull.get_disaggregation_constraint(m.y, m.parent1.disjunction)
    y_cons_expr = self.simplify_cons(y_cons)
    assertExpressionsEqual(self, y_cons_expr, y_p1 - other_y - y_c2 == 0.0)
    y_cons = hull.get_disaggregation_constraint(m.y, m.parent_disjunction)
    y_cons_expr = self.simplify_cons(y_cons)
    assertExpressionsEqual(self, y_cons_expr, m.y - y_p2 - y_p1 == 0.0)
    x_c1 = hull.get_disaggregated_var(m.x, m.child1)
    x_c2 = hull.get_disaggregated_var(m.x, m.child2)
    x_c3 = hull.get_disaggregated_var(m.x, m.child3)
    x_p1 = hull.get_disaggregated_var(m.x, m.parent1)
    x_p2 = hull.get_disaggregated_var(m.x, m.parent2)
    x_cons_parent = hull.get_disaggregation_constraint(m.x, m.parent_disjunction)
    assertExpressionsEqual(self, x_cons_parent.expr, m.x == x_p1 + x_p2)
    x_cons_child = hull.get_disaggregation_constraint(m.x, m.parent1.disjunction)
    x_cons_child_expr = self.simplify_cons(x_cons_child)
    assertExpressionsEqual(self, x_cons_child_expr, x_p1 - x_c1 - x_c2 - x_c3 == 0.0)