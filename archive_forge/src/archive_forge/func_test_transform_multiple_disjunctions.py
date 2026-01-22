from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.common.log import LoggingIntercept
def test_transform_multiple_disjunctions(self):
    m = self.create_two_disjunction_model()
    bt = TransformationFactory('gdp.bound_pretransformation')
    bt.apply_to(m)
    self.check_nested_model_disjunction(m, bt)
    cons = bt.get_transformed_constraints(m.x, m.disjunction)
    self.assertEqual(len(cons), 2)
    lb = cons[0]
    assertExpressionsEqual(self, lb.expr, -100 * m.d1.binary_indicator_var + 34.0 * m.d2.binary_indicator_var + -100 * m.d3.binary_indicator_var <= m.x)
    ub = cons[1]
    assertExpressionsEqual(self, ub.expr, 27.0 * m.d1.binary_indicator_var + 102 * m.d2.binary_indicator_var + 102 * m.d3.binary_indicator_var >= m.x)
    cons = bt.get_transformed_constraints(m.y, m.disjunction)
    self.assertEqual(len(cons), 1)
    ub = cons[0]
    assertExpressionsEqual(self, ub.expr, 7.8 * m.d1.binary_indicator_var + 8.9 * m.d2.binary_indicator_var + 45.7 * m.d3.binary_indicator_var >= m.y)
    self.assertFalse(m.d1.c.active)
    self.assertFalse(m.d1.c_x.active)
    self.assertFalse(m.d2.c.active)
    self.assertFalse(m.d2.c_x.active)
    self.assertFalse(m.d3.c.active)
    c_lb = m.d1.component('c_lb')
    self.assertIsInstance(c_lb, Constraint)
    self.assertTrue(c_lb.active)
    assertExpressionsEqual(self, c_lb.expr, 7.8 <= m.y)
    c_lb = m.d2.component('c_lb')
    self.assertIsInstance(c_lb, Constraint)
    self.assertTrue(c_lb.active)
    assertExpressionsEqual(self, c_lb.expr, 8.9 <= m.y)
    self.assertEqual(len(list(m.component_data_objects(Constraint, active=True, descend_into=(Block, Disjunct)))), 9)