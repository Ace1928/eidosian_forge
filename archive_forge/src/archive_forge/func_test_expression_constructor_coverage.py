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
def test_expression_constructor_coverage(self):
    x = variable()
    y = variable()
    z = variable()
    L = parameter(value=0)
    U = parameter(value=1)
    expr = U >= x
    expr = expr >= L
    c = constraint(expr)
    expr = x <= z
    expr = expr >= y
    with self.assertRaises(ValueError):
        constraint(expr)
    expr = x >= z
    expr = y >= expr
    with self.assertRaises(ValueError):
        constraint(expr)
    expr = y <= x
    expr = y >= expr
    with self.assertRaises(ValueError):
        constraint(expr)
    L.value = 0
    c = constraint(x >= L)
    U.value = 0
    c = constraint(U >= x)
    L.value = 0
    U.value = 1
    expr = U <= x
    expr = expr <= L
    c = constraint(expr)
    expr = x >= z
    expr = expr <= y
    with self.assertRaises(ValueError):
        constraint(expr)
    expr = x <= z
    expr = y <= expr
    with self.assertRaises(ValueError):
        constraint(expr)
    expr = y >= x
    expr = y <= expr
    with self.assertRaises(ValueError):
        constraint(expr)
    L.value = 0
    expr = x <= L
    c = constraint(expr)
    U.value = 0
    expr = U <= x
    c = constraint(expr)
    x = variable()
    with self.assertRaises(ValueError):
        constraint(x + x)