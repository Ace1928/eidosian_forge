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
def test_ordered_keys(self):
    m = ConcreteModel()
    init_keys = [2, 1, (1, 2), (1, 'a'), (1, 1)]
    m.I = Set(ordered=False, dimen=None, initialize=init_keys)
    ordered_keys = [1, 2, (1, 1), (1, 2), (1, 'a')]
    m.x = Var(m.I)
    self.assertNotEqual(list(m.x.keys()), list(m.x.keys(True)))
    self.assertEqual(set(m.x.keys()), set(m.x.keys(True)))
    self.assertEqual(ordered_keys, list(m.x.keys(True)))
    m.P = Param(m.I, initialize={k: v for v, k in enumerate(init_keys)})
    self.assertNotEqual(list(m.P.keys()), list(m.P.keys(True)))
    self.assertEqual(set(m.P.keys()), set(m.P.keys(True)))
    self.assertEqual(ordered_keys, list(m.P.keys(True)))
    self.assertEqual([1, 0, 4, 2, 3], list(m.P.values(True)))
    self.assertEqual(list(zip(ordered_keys, [1, 0, 4, 2, 3])), list(m.P.items(True)))
    m.P = Param(m.I, initialize={(1, 2): 30, 1: 10, 2: 20}, default=1)
    self.assertNotEqual(list(m.P.keys()), list(m.P.keys(True)))
    self.assertEqual(set(m.P.keys()), set(m.P.keys(True)))
    self.assertEqual(ordered_keys, list(m.P.keys(True)))
    self.assertEqual([10, 20, 1, 30, 1], list(m.P.values(True)))
    self.assertEqual(list(zip(ordered_keys, [10, 20, 1, 30, 1])), list(m.P.items(True)))