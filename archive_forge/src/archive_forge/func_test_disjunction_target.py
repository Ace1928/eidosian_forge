from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.common.log import LoggingIntercept
def test_disjunction_target(self):
    m = self.create_two_disjunction_model()
    bt = TransformationFactory('gdp.bound_pretransformation')
    bt.apply_to(m, targets=m.outer)
    self.check_nested_model_disjunction(m, bt)
    self.assertTrue(m.d1.c.active)
    self.assertTrue(m.d1.c_x.active)
    self.assertTrue(m.d2.c.active)
    self.assertTrue(m.d2.c_x.active)
    self.assertTrue(m.d3.c.active)
    self.assertEqual(len(list(m.component_data_objects(Constraint, active=True, descend_into=(Block, Disjunct)))), 9)