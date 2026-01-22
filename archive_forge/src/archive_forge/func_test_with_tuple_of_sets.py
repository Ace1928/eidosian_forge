import pyomo.common.unittest as unittest
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.global_set import UnindexedComponent_set
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.slices import (
def test_with_tuple_of_sets(self):
    m = pyo.ConcreteModel()
    m.s1 = pyo.Set(initialize=[1, 2, 3])
    m.s2 = pyo.Set(initialize=[1, 2, 3])
    m.v = pyo.Var(m.s1, m.s2)
    sets = (m.s1,)
    slice_ = slice_component_along_sets(m.v[1, 2], sets)
    self.assertEqual(str(pyo.ComponentUID(slice_)), 'v[*,2]')
    self.assertEqual(slice_, m.v[:, 2])