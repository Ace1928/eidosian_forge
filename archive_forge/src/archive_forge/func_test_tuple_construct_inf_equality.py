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
def test_tuple_construct_inf_equality(self):
    x = variable()
    c = constraint((x, float('inf')))
    self.assertEqual(c.equality, True)
    self.assertEqual(c.lb, float('inf'))
    self.assertEqual(c.ub, None)
    self.assertEqual(c.rhs, float('inf'))
    self.assertEqual(type(c.rhs), float)
    self.assertIs(c.body, x)
    c = constraint((float('inf'), x))
    self.assertEqual(c.equality, True)
    self.assertEqual(c.lb, float('inf'))
    self.assertEqual(c.ub, None)
    self.assertEqual(c.rhs, float('inf'))
    self.assertEqual(type(c.rhs), float)
    self.assertIs(c.body, x)