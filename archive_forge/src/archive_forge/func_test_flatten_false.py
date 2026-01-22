import pickle
import pyomo.common.unittest as unittest
from pyomo.environ import Var, Block, ConcreteModel, RangeSet, Set, Any
from pyomo.core.base.block import _BlockData
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.set import normalize_index
def test_flatten_false(self):
    _old_flatten = normalize_index.flatten
    try:
        normalize_index.flatten = False
        m = ConcreteModel()
        m.I = Set(initialize=range(2))
        m.J = Set(initialize=range(2, 4))
        m.K = Set(initialize=['a', 'b', 'c'])
        m.IJ = m.I * m.J
        m.a = Var(m.I, m.J, m.K)
        m.b = Var(m.IJ, m.K)
        m.c = Var()
        with self.assertRaisesRegex(IndexError, 'Index .* contains an invalid number of entries for component .*'):
            _slicer = m.a[(0, 2), :]
        _slicer = m.a[0, 2, :]
        names = ['a[0,2,a]', 'a[0,2,b]', 'a[0,2,c]']
        self.assertEqual(names, [var.name for var in _slicer])
        with self.assertRaisesRegex(IndexError, 'Index .* contains an invalid number of entries for component .*'):
            _slicer = m.b[0, 2, :]
        _slicer = m.b[(0, 2), :]
        names = ['b[(0,2),a]', 'b[(0,2),b]', 'b[(0,2),c]']
        self.assertEqual(names, [var.name for var in _slicer])
        with self.assertRaisesRegex(IndexError, 'Index .* contains an invalid number of entries for component .*'):
            _slicer = m.b[:, 2, 'b']
        _slicer = m.b[:, 'b']
        names = ['b[(0,2),b]', 'b[(0,3),b]', 'b[(1,2),b]', 'b[(1,3),b]']
        self.assertEqual(names, [var.name for var in _slicer])
        _slicer = m.b[..., 'b']
        self.assertEqual(names, [var.name for var in _slicer])
        _slicer = m.b[0, ...]
        self.assertEqual([], [var.name for var in _slicer])
        _slicer = m.c[:]
        self.assertEqual(['c'], [var.name for var in _slicer])
    finally:
        normalize_index.flatten = _old_flatten