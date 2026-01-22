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
def test_expr_construct_1sided_inequality(self):
    y = variable()
    c = constraint(y <= 1)
    self.assertEqual(c.equality, False)
    self.assertIs(c.lb, None)
    self.assertIs(c.body, y)
    self.assertEqual(c.ub, 1)
    c = constraint(0 <= y)
    self.assertEqual(c.equality, False)
    self.assertEqual(c.lb, 0)
    self.assertIs(c.body, y)
    self.assertIs(c.ub, None)
    c = constraint(y >= 1)
    self.assertEqual(c.equality, False)
    self.assertEqual(c.lb, 1)
    self.assertIs(c.body, y)
    self.assertIs(c.ub, None)
    c = constraint(0 >= y)
    self.assertEqual(c.equality, False)
    self.assertIs(c.lb, None)
    self.assertIs(c.body, y)
    self.assertEqual(c.ub, 0)