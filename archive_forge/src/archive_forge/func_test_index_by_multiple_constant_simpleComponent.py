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
def test_index_by_multiple_constant_simpleComponent(self):
    m = ConcreteModel()
    m.i = Param(initialize=2)
    m.j = Param(initialize=3)
    m.x = Var([1, 2, 3], [1, 2, 3], initialize=lambda m, x, y: 2 * x * y)
    self.assertEqual(value(m.x[2, 3]), 12)
    self.assertEqual(value(m.x[m.i, 3]), 12)
    self.assertEqual(value(m.x[m.i, m.j]), 12)
    self.assertEqual(value(m.x[2, m.j]), 12)
    self.assertIs(m.x[2, 3], m.x[m.i, 3])
    self.assertIs(m.x[2, 3], m.x[m.i, m.j])
    self.assertIs(m.x[2, 3], m.x[2, m.j])