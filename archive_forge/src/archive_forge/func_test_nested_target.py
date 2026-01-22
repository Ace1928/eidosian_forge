from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.common.log import LoggingIntercept
def test_nested_target(self):
    m = self.create_nested_model()
    bt = TransformationFactory('gdp.bound_pretransformation')
    bt.apply_to(m, targets=[m.outer_d1.inner])
    cons = bt.get_transformed_constraints(m.x, m.outer_d1.inner)
    self.assertEqual(len(cons), 2)
    lb = cons[0]
    ub = cons[1]
    assertExpressionsEqual(self, lb.expr, -100 * m.outer_d1.inner_d1.binary_indicator_var - 7.0 * m.outer_d1.inner_d2.binary_indicator_var <= m.x)
    self.assertIs(lb.parent_block().parent_block(), m.outer_d1)
    assertExpressionsEqual(self, ub.expr, 3.0 * m.outer_d1.inner_d1.binary_indicator_var + 102 * m.outer_d1.inner_d2.binary_indicator_var >= m.x)
    self.assertIs(ub.parent_block().parent_block(), m.outer_d1)
    self.assertTrue(m.outer_d1.c.active)
    self.assertTrue(m.outer_d2.c.active)
    self.assertTrue(lb.active)
    self.assertTrue(ub.active)
    self.assertEqual(len(list(m.component_data_objects(Constraint, active=True, descend_into=(Block, Disjunct)))), 4)