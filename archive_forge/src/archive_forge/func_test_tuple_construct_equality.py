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
def test_tuple_construct_equality(self):
    x = variable()
    c = constraint((0.0, x))
    self.assertEqual(c.equality, True)
    self.assertEqual(c.lb, 0)
    self.assertEqual(type(c.lb), float)
    self.assertIs(c.body, x)
    self.assertEqual(c.ub, 0)
    self.assertEqual(type(c.ub), float)
    c = constraint((x, 0))
    self.assertEqual(c.equality, True)
    self.assertEqual(c.lb, 0)
    self.assertEqual(type(c.lb), int)
    self.assertIs(c.body, x)
    self.assertEqual(c.ub, 0)
    self.assertEqual(type(c.ub), int)