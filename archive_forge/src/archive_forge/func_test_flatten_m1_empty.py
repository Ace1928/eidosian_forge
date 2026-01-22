import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_flatten_m1_empty(self):
    m = self._model1_1d_sets()
    sets = ComponentSet()
    sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
    assert len(sets_list) == len(comps_list)
    assert len(sets_list) == 1
    for sets, comps in zip(sets_list, comps_list):
        if len(sets) == 1 and sets[0] is UnindexedComponent_set:
            ref_data = {self._hashRef(v) for v in m.component_data_objects(Var)}
            assert len(comps) == len(ref_data)
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        else:
            raise RuntimeError()