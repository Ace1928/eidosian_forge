import pyomo.common.unittest as unittest
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.global_set import UnindexedComponent_set
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.slices import (
def test_none_dimen(self):
    m = self.model()
    comp = m.b.b2[1, 0].vn
    sets = ComponentSet((m.d_none,))
    _slice = slice_component_along_sets(comp, sets)
    self.assertIs(_slice, m.b.b2[1, 0].vn)
    comp = m.b.b2[1, 0].vn[1, 'd', 3, 'a', 1]
    sets = ComponentSet((m.d_2, m.time))
    _slice = slice_component_along_sets(comp, sets)
    self.assertEqual(_slice, m.b.b2[:, 0].vn[:, 'd', 3, :, :])
    sets = ComponentSet((m.d_none, m.d_2))
    _slice = slice_component_along_sets(comp, sets)
    self.assertEqual(_slice, m.b.b2[1, 0].vn[1, ..., :, :])
    comp = m.b.bn['c', 1, 10, 'a', 1].v3[1, 0, 1]
    sets = ComponentSet((m.d_none, m.time))
    _slice = slice_component_along_sets(comp, sets)
    self.assertEqual(_slice, m.b.bn[..., 'a', 1].v3[:, 0, :])
    comp = m.b.bn['c', 1, 10, 'a', 1].vn[1, 'd', 3, 'b', 2]
    sets = ComponentSet((m.d_none,))
    _slice = slice_component_along_sets(comp, sets)
    self.assertEqual(_slice, m.b.bn[..., 'a', 1].vn[1, ..., 'b', 2])
    sets = ComponentSet((m.d_none, m.d_2))
    _slice = slice_component_along_sets(comp, sets)
    self.assertEqual(_slice, m.b.bn[..., :, :].vn[1, ..., :, :])
    comp = m.b.bn['c', 1, 10, 'a', 1].vn[1, 'd', 3, 'b', 2]
    context = m.b.bn['c', 1, 10, 'a', 1]
    sets = ComponentSet((m.d_none,))
    _slice = slice_component_along_sets(comp, sets, context=context)
    self.assertEqual(_slice, m.b.bn['c', 1, 10, 'a', 1].vn[1, ..., 'b', 2])
    sets = ComponentSet((m.d_none, m.d_2))
    _slice = slice_component_along_sets(comp, sets, context=context)
    self.assertEqual(_slice, m.b.bn['c', 1, 10, 'a', 1].vn[1, ..., :, :])
    context = m.b.bn
    sets = ComponentSet((m.d_none,))
    _slice = slice_component_along_sets(comp, sets, context=context)
    self.assertEqual(_slice, m.b.bn[..., 'a', 1].vn[1, ..., 'b', 2])