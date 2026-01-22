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
def test_tuple_construct_invalid_2sided_inequality(self):
    x = variable()
    y = variable()
    z = variable()
    with self.assertRaises(ValueError):
        constraint(RangedExpression((x, y, 1), (False, False)))
    with self.assertRaises(ValueError):
        constraint(RangedExpression((0, y, z), (False, False)))