import pickle
import pyomo.common.unittest as unittest
from pyomo.environ import Var, Block, ConcreteModel, RangeSet, Set, Any
from pyomo.core.base.block import _BlockData
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.set import normalize_index
def test_setitem_component(self):
    init_sum = sum(self.m.x[:].value)
    init_vals = list(self.m.x[:].value)
    self.m.x[:] = 0
    new_sum = sum(self.m.x[:].value)
    new_vals = list(self.m.x[:].value)
    self.assertEqual(len(init_vals), len(new_vals))
    self.assertNotEqual(init_vals, new_vals)
    self.assertEqual(sum(new_vals), 0)
    self.assertEqual(init_sum - sum(init_vals), new_sum)
    init_sum = sum(self.m.y[:, :].value)
    init_vals = list(self.m.y[1, :].value)
    self.m.y[1, :] = 0
    new_sum = sum(self.m.y[:, :].value)
    new_vals = list(self.m.y[1, :].value)
    self.assertEqual(len(init_vals), len(new_vals))
    self.assertNotEqual(init_vals, new_vals)
    self.assertEqual(sum(new_vals), 0)
    self.assertEqual(init_sum - sum(init_vals), new_sum)