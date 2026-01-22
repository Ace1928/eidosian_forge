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
def test_expr_construct_equality(self):
    x = variable(value=1)
    y = variable(value=1)
    c = constraint(0.0 == x)
    self.assertEqual(c.equality, True)
    self.assertEqual(c.lb, 0)
    self.assertIs(c.body, x)
    self.assertEqual(c.ub, 0)
    c = constraint(x == 0.0)
    self.assertEqual(c.equality, True)
    self.assertEqual(c.lb, 0)
    self.assertIs(c.body, x)
    self.assertEqual(c.ub, 0)
    c = constraint(x == y)
    self.assertEqual(c.equality, True)
    self.assertEqual(c.lb, 0)
    self.assertTrue(c.body is not None)
    self.assertEqual(c(), 0)
    self.assertEqual(c.body(), 0)
    self.assertEqual(c.ub, 0)
    c = constraint()
    c.expr = x == float('inf')
    self.assertEqual(c.equality, True)
    self.assertEqual(c.lb, float('inf'))
    self.assertEqual(c.ub, None)
    self.assertEqual(c.rhs, float('inf'))
    self.assertIs(c.body, x)
    c.expr = float('inf') == x
    self.assertEqual(c.equality, True)
    self.assertEqual(c.lb, float('inf'))
    self.assertEqual(c.ub, None)
    self.assertEqual(c.rhs, float('inf'))
    self.assertIs(c.body, x)