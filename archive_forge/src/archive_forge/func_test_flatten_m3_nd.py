import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_flatten_m3_nd(self):
    m = self._model3_nd_sets_normalizeflatten()
    m.del_component(m.v_1n2n)
    sets = ComponentSet((m.dn,))
    sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
    assert len(sets_list) == len(comps_list)
    assert len(sets_list) == 3
    for sets, comps in zip(sets_list, comps_list):
        if len(sets) == 1 and sets[0] is UnindexedComponent_set:
            ref_data = set()
            ref_data.update((self._hashRef(v) for v in m.v_12.values()))
            ref_data.update((self._hashRef(v) for v in m.v_212.values()))
            assert len(comps) == len(ref_data)
            assert len(comps) == 12
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        elif len(sets) == 1 and sets[0] is m.dn:
            ref_data = set()
            ref_data.update((self._hashRef(Reference(m.v_2n[i2, ...])) for i2 in m.d2))
            ref_data.update((self._hashRef(Reference(m.v_12n[i1, i2, ...])) for i1 in m.d1 for i2 in m.d2))
            ref_data.update((self._hashRef(Reference(m.b[i1, i2, ...].v0)) for i1 in m.d1 for i2 in m.d2))
            ref_data.update((self._hashRef(Reference(m.b[i1a, i2, ...].v1[i1b])) for i1a in m.d1 for i2 in m.d2 for i1b in m.d1))
            ref_data.update((self._hashRef(Reference(m.b[i1, i2a, ...].v2[i2b])) for i1 in m.d1 for i2a in m.d2 for i2b in m.d2))
            assert len(comps) == len(ref_data)
            assert len(comps) == 26
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        elif len(sets) == 2 and sets[0] is m.dn and (sets[1] is m.dn):
            ref_data = set()
            ref_data.update((self._hashRef(Reference(m.b[i1, i2, ...].vn[...])) for i1 in m.d1 for i2 in m.d2))
            assert len(comps) == len(ref_data)
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        else:
            raise RuntimeError()