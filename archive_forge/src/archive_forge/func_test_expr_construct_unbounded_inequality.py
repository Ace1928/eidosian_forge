import pickle
import pyomo.common.unittest as unittest
from pyomo.core.expr import inequality, RangedExpression, EqualityExpression
from pyomo.kernel import pprint
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_tuple_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.variable import variable
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.expression import expression, data_expression
from pyomo.core.kernel.block import block
def test_expr_construct_unbounded_inequality(self):
    y = variable()
    c = constraint(y <= float('-inf'))
    self.assertEqual(c.equality, False)
    self.assertIs(c.lb, None)
    self.assertEqual(c.ub, float('-inf'))
    self.assertIs(c.body, y)
    c = constraint(float('inf') <= y)
    self.assertEqual(c.equality, False)
    self.assertEqual(c.lb, float('inf'))
    self.assertIs(c.ub, None)
    self.assertIs(c.body, y)
    c = constraint(y >= float('inf'))
    self.assertEqual(c.equality, False)
    self.assertEqual(c.lb, float('inf'))
    self.assertIs(c.ub, None)
    self.assertIs(c.body, y)
    c = constraint(float('-inf') >= y)
    self.assertEqual(c.equality, False)
    self.assertIs(c.lb, None)
    self.assertEqual(c.ub, float('-inf'))
    self.assertIs(c.body, y)