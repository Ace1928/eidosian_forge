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
def test_init_nonexpr(self):
    v = variable(value=3)
    c = linear_constraint()
    self.assertEqual(len(list(c.terms)), 0)
    self.assertEqual(c.lb, None)
    self.assertEqual(c.body, 0)
    self.assertEqual(c.ub, None)
    c = linear_constraint([v], [1], lb=0, ub=1)
    self.assertEqual(len(list(c.terms)), 1)
    self.assertEqual(c.lb, 0)
    self.assertEqual(c.body(), 3)
    self.assertEqual(c(), 3)
    self.assertEqual(c.ub, 1)
    with self.assertRaises(ValueError):
        linear_constraint(terms=(), variables=())
    with self.assertRaises(ValueError):
        linear_constraint(terms=(), coefficients=())
    with self.assertRaises(ValueError):
        linear_constraint(terms=(), variables=(), coefficients=())
    with self.assertRaises(ValueError):
        linear_constraint(variables=[v])
    with self.assertRaises(ValueError):
        linear_constraint(coefficients=[1])
    with self.assertRaises(ValueError):
        linear_constraint([v], [1], lb=0, rhs=0)
    with self.assertRaises(ValueError):
        linear_constraint([v], [1], ub=0, rhs=0)
    c = linear_constraint([v], [1], rhs=1)
    self.assertEqual(c.lb, 1)
    self.assertEqual(c.ub, 1)
    self.assertEqual(c.rhs, 1)
    self.assertEqual(c.body(), 3)
    self.assertEqual(c(), 3)
    c = linear_constraint([], [], rhs=1)
    c.terms = ((v, 1),)
    self.assertEqual(c.lb, 1)
    self.assertEqual(c.ub, 1)
    self.assertEqual(c.rhs, 1)
    self.assertEqual(c.body(), 3)
    self.assertEqual(c(), 3)