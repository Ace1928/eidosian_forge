import pickle
import pyomo.common.unittest as unittest
from pyomo.environ import Var, Block, ConcreteModel, RangeSet, Set, Any
from pyomo.core.base.block import _BlockData
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.set import normalize_index
def test_noncomponent_function_slices(self):
    ans = self.m.component('b')[1, :].component('c')[:, 4].x.fix(5)
    self.assertIsInstance(ans, list)
    self.assertEqual(ans, [None] * 9)
    ans = self.m.component('b')[1, :].component('c')[:, 4].x[:].is_fixed()
    self.assertIsInstance(ans, list)
    self.assertEqual(ans, [True] * (9 * 3))
    ans = self.m.component('b')[1, :].component('c')[:, 5].x[:].is_fixed()
    self.assertIsInstance(ans, list)
    self.assertEqual(ans, [False] * (9 * 3))