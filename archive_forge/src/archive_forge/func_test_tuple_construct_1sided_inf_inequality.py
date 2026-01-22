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
def test_tuple_construct_1sided_inf_inequality(self):
    y = variable()
    c = constraint((float('-inf'), y, 1))
    self.assertEqual(c.equality, False)
    self.assertEqual(c.lb, None)
    self.assertIs(c.body, y)
    self.assertEqual(c.ub, 1)
    self.assertEqual(type(c.ub), int)
    c = constraint((0, y, float('inf')))
    self.assertEqual(c.equality, False)
    self.assertEqual(c.lb, 0)
    self.assertEqual(type(c.lb), int)
    self.assertIs(c.body, y)
    self.assertEqual(c.ub, None)