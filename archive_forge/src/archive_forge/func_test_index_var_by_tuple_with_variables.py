import os
from os.path import abspath, dirname
from pyomo.common import DeveloperError
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.environ import ConcreteModel, Var, Param, Set, value, Integers
from pyomo.core.base.set import FiniteSetOf, OrderedSetOf
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.expr import GetItemExpression
from pyomo.core import SortComponents
def test_index_var_by_tuple_with_variables(self):
    m = ConcreteModel()
    m.x = Var([(1, 1), (2, 1), (1, 2), (2, 2)])
    m.i = Var([1, 2, 3], domain=Integers)
    thing = m.x[1, m.i[1]]
    self.assertIsInstance(thing, GetItemExpression)
    self.assertEqual(len(thing.args), 3)
    self.assertIs(thing.args[0], m.x)
    self.assertEqual(thing.args[1], 1)
    self.assertIs(thing.args[2], m.i[1])
    idx_expr = m.i[1] + m.i[2] * m.i[3]
    thing = m.x[1, idx_expr]
    self.assertIsInstance(thing, GetItemExpression)
    self.assertEqual(len(thing.args), 3)
    self.assertIs(thing.args[0], m.x)
    self.assertEqual(thing.args[1], 1)
    self.assertIs(thing.args[2], idx_expr)