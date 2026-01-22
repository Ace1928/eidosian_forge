import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_deactivated_block_active_true(self):
    m = self._model1_1d_sets()
    m.b.b1.deactivate()
    sets = (m.time,)
    sets_list, comps_list = flatten_components_along_sets(m, sets, Var, active=True)
    expected_unindexed = [ComponentUID(m.v0)]
    expected_unindexed = set(expected_unindexed)
    expected_time = [ComponentUID(m.v1[:])]
    expected_time.extend((ComponentUID(m.v2[:, x]) for x in m.space))
    expected_time.extend((ComponentUID(m.v3[:, x, j]) for x in m.space for j in m.comp))
    expected_time.extend((ComponentUID(m.b.b2[:, x].v0) for x in m.space))
    expected_time.extend((ComponentUID(m.b.b2[:, x].v1[j]) for x in m.space for j in m.comp))
    expected_time = set(expected_time)
    expected_2time = [ComponentUID(m.v_tt[:, :])]
    expected_2time.extend((ComponentUID(m.v_tst[:, x, :]) for x in m.space))
    expected_2time.extend((ComponentUID(m.b.b2[:, x].v2[:, j]) for x in m.space for j in m.comp))
    expected_2time = set(expected_2time)
    set_id_set = set((tuple((id(s) for s in sets)) for sets in sets_list))
    pred_sets = [(UnindexedComponent_set,), (m.time,), (m.time, m.time)]
    pred_set_ids = set((tuple((id(s) for s in sets)) for sets in pred_sets))
    self.assertEqual(set_id_set, pred_set_ids)
    for sets, comps in zip(sets_list, comps_list):
        if len(sets) == 1 and sets[0] is UnindexedComponent_set:
            comp_set = set((ComponentUID(comp) for comp in comps))
            self.assertEqual(comp_set, expected_unindexed)
        elif len(sets) == 1 and sets[0] is m.time:
            comp_set = set((ComponentUID(comp.referent) for comp in comps))
            self.assertEqual(comp_set, expected_time)
        elif len(sets) == 2:
            self.assertIs(sets[0], m.time)
            self.assertIs(sets[1], m.time)
            comp_set = set((ComponentUID(comp.referent) for comp in comps))
            self.assertEqual(comp_set, expected_2time)