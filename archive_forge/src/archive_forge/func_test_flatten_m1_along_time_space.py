import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_flatten_m1_along_time_space(self):
    m = self._model1_1d_sets()
    sets = ComponentSet((m.time, m.space))
    sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
    assert len(sets_list) == len(comps_list)
    assert len(sets_list) == 6
    for sets, comps in zip(sets_list, comps_list):
        if len(sets) == 1 and sets[0] is UnindexedComponent_set:
            ref_data = {self._hashRef(m.v0)}
            assert len(comps) == len(ref_data)
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        elif len(sets) == 1 and sets[0] is m.time:
            ref_data = {self._hashRef(Reference(m.v1)), self._hashRef(Reference(m.b.b1[:].v0))}
            assert len(comps) == len(ref_data)
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        elif len(sets) == 2 and sets[0] is m.time and (sets[1] is m.time):
            ref_data = {self._hashRef(Reference(m.v_tt))}
            assert len(comps) == len(ref_data)
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        elif len(sets) == 2 and sets[0] is m.time and (sets[1] is m.space):
            ref_data = {self._hashRef(m.v2), self._hashRef(Reference(m.b.b1[:].v1[:])), self._hashRef(Reference(m.b.b2[:, :].v0)), self._hashRef(Reference(m.b.b1[:].b_s[:].v0))}
            ref_data.update((self._hashRef(Reference(m.v3[:, :, j])) for j in m.comp))
            ref_data.update((self._hashRef(Reference(m.b.b1[:].v2[:, j])) for j in m.comp))
            ref_data.update((self._hashRef(Reference(m.b.b2[:, :].v1[j])) for j in m.comp))
            assert len(comps) == len(ref_data)
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        elif len(sets) == 3 and sets[0] is m.time and (sets[1] is m.space) and (sets[2] is m.time):
            ref_data = {self._hashRef(m.v_tst)}
            ref_data.update((self._hashRef(Reference(m.b.b2[:, :].v2[:, j])) for j in m.comp))
            assert len(comps) == len(ref_data)
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        elif len(sets) == 3 and sets[0] is m.time and (sets[1] is m.space) and (sets[2] is m.space):
            ref_data = {self._hashRef(Reference(m.b.b1[:].b_s[:].v1[:]))}
            (ref_data.update((self._hashRef(Reference(m.b.b1[:].b_s[:].v2[:, j])) for j in m.comp)),)
            assert len(comps) == len(ref_data)
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        else:
            raise RuntimeError()