import pickle
import pyomo.common.unittest as unittest
from pyomo.environ import Var, Block, ConcreteModel, RangeSet, Set, Any
from pyomo.core.base.block import _BlockData
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.set import normalize_index
def test_nondim_set(self):
    m = ConcreteModel()
    m.I = Set(dimen=None, initialize=[1, (2, 3)])
    m.x = Var(m.I)
    ref = list(m.x[:])
    self.assertEqual(len(ref), 1)
    self.assertIs(ref[0], m.x[1])
    ref = list(m.x[:, ..., :])
    self.assertEqual(len(ref), 1)
    self.assertIs(ref[0], m.x[2, 3])
    ref = list(m.x[2, ...])
    self.assertEqual(len(ref), 1)
    self.assertIs(ref[0], m.x[2, 3])
    _old_flatten = normalize_index.flatten
    try:
        normalize_index.flatten = False
        m = ConcreteModel()
        m.I = Set(dimen=None, initialize=[1, (2, 3)])
        m.x = Var(m.I)
        ref = list(m.x[:])
        self.assertEqual(len(ref), 2)
        self.assertIs(ref[0], m.x[1])
        with self.assertRaisesRegex(IndexError, 'Index .* contains an invalid number of entries for component .*'):
            list(m.x[:, ..., :])
    finally:
        normalize_index.flatten = _old_flatten