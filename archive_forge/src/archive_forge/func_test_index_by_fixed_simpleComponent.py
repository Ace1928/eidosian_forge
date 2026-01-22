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
def test_index_by_fixed_simpleComponent(self):
    m = ConcreteModel()
    m.i = Param(initialize=2, mutable=True)
    m.x = Var([1, 2, 3], initialize=lambda m, x: 2 * x)
    self.assertEqual(value(m.x[2]), 4)
    self.assertRaisesRegex(RuntimeError, 'is a fixed but not constant value', m.x.__getitem__, m.i)