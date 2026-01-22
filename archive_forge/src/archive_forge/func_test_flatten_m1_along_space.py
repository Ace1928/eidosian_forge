import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_flatten_m1_along_space(self):
    m = self._model1_1d_sets()
    sets = ComponentSet((m.space,))
    sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
    assert len(sets_list) == len(comps_list)
    assert len(sets_list) == 3
    T = m.time
    TC = m.time * m.comp
    TT = m.time * m.time
    TTC = m.time * m.time * m.comp
    for sets, comps in zip(sets_list, comps_list):
        if len(sets) == 1 and sets[0] is UnindexedComponent_set:
            ref_data = {self._hashRef(m.v0)}
            ref_data.update((self._hashRef(m.v1[t]) for t in T))
            ref_data.update((self._hashRef(m.v_tt[t1, t2]) for t1, t2 in TT))
            ref_data.update((self._hashRef(m.b.b1[t].v0) for t in T))
            assert len(comps) == len(ref_data)
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        elif len(sets) == 1 and sets[0] is m.space:
            ref_data = set()
            ref_data.update((self._hashRef(Reference(m.v2[t, :])) for t in T))
            ref_data.update((self._hashRef(Reference(m.v3[t, :, j])) for t, j in TC))
            ref_data.update((self._hashRef(Reference(m.v_tst[t1, :, t2])) for t1, t2 in TT))
            ref_data.update((self._hashRef(Reference(m.b.b1[t].v1[:])) for t in T))
            ref_data.update((self._hashRef(Reference(m.b.b1[t].v2[:, j])) for t, j in TC))
            ref_data.update((self._hashRef(Reference(m.b.b1[t].b_s[:].v0)) for t in T))
            ref_data.update((self._hashRef(Reference(m.b.b2[t, :].v0)) for t in T))
            ref_data.update((self._hashRef(Reference(m.b.b2[t, :].v1[j])) for t, j in TC))
            ref_data.update((self._hashRef(Reference(m.b.b2[t1, :].v2[t2, j])) for t1, t2, j in TTC))
            assert len(comps) == len(ref_data)
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        elif len(sets) == 2 and sets[0] is m.space and (sets[1] is m.space):
            ref_data = set()
            ref_data.update((self._hashRef(Reference(m.b.b1[t].b_s[:].v1[:])) for t in T))
            ref_data.update((self._hashRef(Reference(m.b.b1[t].b_s[:].v2[:, j])) for t, j in TC))
            assert len(comps) == len(ref_data)
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        else:
            raise RuntimeError()