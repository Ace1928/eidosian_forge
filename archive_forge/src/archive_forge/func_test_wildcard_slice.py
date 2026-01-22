import pickle
import pyomo.common.unittest as unittest
from pyomo.environ import Var, Block, ConcreteModel, RangeSet, Set, Any
from pyomo.core.base.block import _BlockData
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.set import normalize_index
def test_wildcard_slice(self):
    with self.assertRaisesRegex(IndexError, 'Index .* contains an invalid number of entries for component .*'):
        _slicer = self.m.b[:]
    _slicer = self.m.b[...]
    self.assertIsInstance(_slicer, IndexedComponent_slice)
    ans = [str(x) for x in _slicer]
    self.assertEqual(ans, ['b[1,4]', 'b[1,5]', 'b[1,6]', 'b[2,4]', 'b[2,5]', 'b[2,6]', 'b[3,4]', 'b[3,5]', 'b[3,6]'])
    _slicer = self.m.b[1, ...]
    self.assertIsInstance(_slicer, IndexedComponent_slice)
    ans = [str(x) for x in _slicer]
    self.assertEqual(ans, ['b[1,4]', 'b[1,5]', 'b[1,6]'])
    _slicer = self.m.b[..., 5]
    self.assertIsInstance(_slicer, IndexedComponent_slice)
    ans = [str(x) for x in _slicer]
    self.assertEqual(ans, ['b[1,5]', 'b[2,5]', 'b[3,5]'])
    _slicer = self.m.bb[2, ..., 8]
    self.assertIsInstance(_slicer, IndexedComponent_slice)
    ans = [str(x) for x in _slicer]
    self.assertEqual(ans, ['bb[2,4,8]', 'bb[2,5,8]', 'bb[2,6,8]'])
    _slicer = self.m.bb[:, ..., 8]
    self.assertIsInstance(_slicer, IndexedComponent_slice)
    ans = [str(x) for x in _slicer]
    self.assertEqual(ans, ['bb[1,4,8]', 'bb[1,5,8]', 'bb[1,6,8]', 'bb[2,4,8]', 'bb[2,5,8]', 'bb[2,6,8]', 'bb[3,4,8]', 'bb[3,5,8]', 'bb[3,6,8]'])
    _slicer = self.m.bb[:, :, ..., 8]
    self.assertIsInstance(_slicer, IndexedComponent_slice)
    ans = [str(x) for x in _slicer]
    self.assertEqual(ans, ['bb[1,4,8]', 'bb[1,5,8]', 'bb[1,6,8]', 'bb[2,4,8]', 'bb[2,5,8]', 'bb[2,6,8]', 'bb[3,4,8]', 'bb[3,5,8]', 'bb[3,6,8]'])
    _slicer = self.m.bb[:, ..., :, 8]
    self.assertIsInstance(_slicer, IndexedComponent_slice)
    ans = [str(x) for x in _slicer]
    self.assertEqual(ans, ['bb[1,4,8]', 'bb[1,5,8]', 'bb[1,6,8]', 'bb[2,4,8]', 'bb[2,5,8]', 'bb[2,6,8]', 'bb[3,4,8]', 'bb[3,5,8]', 'bb[3,6,8]'])
    _slicer = self.m.b[1, 4, ...]
    self.assertIsInstance(_slicer, IndexedComponent_slice)
    ans = [str(x) for x in _slicer]
    self.assertEqual(ans, ['b[1,4]'])
    with self.assertRaisesRegex(IndexError, 'Index .* contains an invalid number of entries for component .*'):
        _slicer = self.m.b[1, 2, 3, ...]
    with self.assertRaisesRegex(IndexError, 'Index .* contains an invalid number of entries for component .*'):
        _slicer = self.m.b[1, :, 2]
    self.assertRaisesRegex(IndexError, 'wildcard slice .* can only appear once', self.m.b.__getitem__, (Ellipsis, Ellipsis))