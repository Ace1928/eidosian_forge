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
def test_equality_infinite(self):
    c = constraint()
    v = variable()
    c.expr = v == 1
    self.assertEqual(c.equality, True)
    self.assertEqual(c.lb, 1)
    self.assertEqual(c.ub, 1)
    self.assertEqual(c.rhs, 1)
    self.assertIs(c.body, v)
    c.expr = v == float('inf')
    self.assertEqual(c.equality, True)
    self.assertEqual(c.lb, float('inf'))
    self.assertEqual(c.ub, None)
    self.assertEqual(c.rhs, float('inf'))
    self.assertIs(c.body, v)
    c.expr = (v, float('inf'))
    self.assertEqual(c.equality, True)
    self.assertEqual(c.lb, float('inf'))
    self.assertEqual(c.ub, None)
    self.assertEqual(c.rhs, float('inf'))
    self.assertIs(c.body, v)
    c.expr = float('inf') == v
    self.assertEqual(c.equality, True)
    self.assertEqual(c.lb, float('inf'))
    self.assertEqual(c.ub, None)
    self.assertEqual(c.rhs, float('inf'))
    self.assertIs(c.body, v)
    c.expr = (float('inf'), v)
    self.assertEqual(c.equality, True)
    self.assertEqual(c.lb, float('inf'))
    self.assertEqual(c.ub, None)
    self.assertEqual(c.rhs, float('inf'))
    self.assertIs(c.body, v)
    c.expr = v == float('-inf')
    self.assertEqual(c.equality, True)
    self.assertEqual(c.lb, None)
    self.assertEqual(c.ub, float('-inf'))
    self.assertEqual(c.rhs, float('-inf'))
    self.assertIs(c.body, v)
    c.expr = (v, float('-inf'))
    self.assertEqual(c.equality, True)
    self.assertEqual(c.lb, None)
    self.assertEqual(c.ub, float('-inf'))
    self.assertEqual(c.rhs, float('-inf'))
    self.assertIs(c.body, v)
    c.expr = float('-inf') == v
    self.assertEqual(c.equality, True)
    self.assertEqual(c.lb, None)
    self.assertEqual(c.ub, float('-inf'))
    self.assertEqual(c.rhs, float('-inf'))
    self.assertIs(c.body, v)
    c.expr = (float('-inf'), v)
    self.assertEqual(c.equality, True)
    self.assertEqual(c.lb, None)
    self.assertEqual(c.ub, float('-inf'))
    self.assertEqual(c.rhs, float('-inf'))
    self.assertIs(c.body, v)