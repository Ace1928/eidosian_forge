from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.common.log import LoggingIntercept
def test_transformation_gives_up_without_enough_bound_info(self):
    """
        If we have unbounded variables and not enough bounding constraints,
        we want the transformation to just leave the bounding constraints
        be to be transformed later.
        """
    m = self.create_nested_structure()
    m.x = Var()
    m.y = Var(bounds=(4, 67))
    m.outer_d1.c = Constraint(Any)
    m.outer_d1.c[1] = m.x >= 3
    m.outer_d1.c[2] = 5 <= m.y
    m.outer_d1.inner_d1.c = Constraint(Any)
    m.outer_d1.inner_d1.c[1] = m.x >= 4
    m.outer_d1.inner_d2.c = Constraint(Any)
    m.outer_d1.inner_d2.c[1] = m.x >= 17
    m.outer_d2.c = Constraint(Any)
    m.outer_d2.c[1] = m.x <= 1
    m.outer_d2.c[2] = m.y <= 66
    m.outer_d2.c[3] = m.x >= 2
    bt = TransformationFactory('gdp.bound_pretransformation')
    bt.apply_to(m)
    cons = bt.get_transformed_constraints(m.x, m.outer)
    self.assertEqual(len(cons), 1)
    lb = cons[0]
    assertExpressionsEqual(self, lb.expr, 3.0 * m.outer_d1.binary_indicator_var + 2.0 * m.outer_d2.binary_indicator_var <= m.x)
    cons = bt.get_transformed_constraints(m.y, m.outer)
    self.assertEqual(len(cons), 2)
    lb = cons[0]
    assertExpressionsEqual(self, lb.expr, 5.0 * m.outer_d1.binary_indicator_var + 4 * m.outer_d2.binary_indicator_var <= m.y)
    ub = cons[1]
    assertExpressionsEqual(self, ub.expr, 67 * m.outer_d1.binary_indicator_var + 66.0 * m.outer_d2.binary_indicator_var >= m.y)
    cons = bt.get_transformed_constraints(m.x, m.outer_d1.inner)
    self.assertEqual(len(cons), 1)
    lb = cons[0]
    assertExpressionsEqual(self, lb.expr, 4.0 * m.outer_d1.inner_d1.binary_indicator_var + 17.0 * m.outer_d1.inner_d2.binary_indicator_var <= m.x)
    self.assertIs(_parent_disjunct(cons[0]), m.outer_d1)
    self.assertFalse(m.outer_d1.c[1].active)
    self.assertFalse(m.outer_d1.c[2].active)
    self.assertFalse(m.outer_d1.inner_d1.c[1].active)
    self.assertFalse(m.outer_d1.inner_d2.c[1].active)
    self.assertTrue(m.outer_d2.c[1].active)
    self.assertFalse(m.outer_d2.c[2].active)
    self.assertFalse(m.outer_d2.c[3].active)
    self.assertEqual(len(list(m.component_data_objects(Constraint, active=True, descend_into=(Block, Disjunct)))), 5)