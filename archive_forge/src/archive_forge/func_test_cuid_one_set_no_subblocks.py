import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_cuid_one_set_no_subblocks(self):
    m = ConcreteModel()
    m.s1 = Set(initialize=[1, 2, 3])
    m.v = Var(m.s1)
    pred_cuid_set = {'v[*]'}
    sets = (m.s1,)
    ctype = Var
    sets_list, comps_list = flatten_components_along_sets(m, sets, ctype)
    self.assertEqual(len(sets_list), 1)
    self.assertEqual(len(comps_list), 1)
    for sets, comps in zip(sets_list, comps_list):
        if len(sets) == 1 and sets[0] is m.s1:
            self.assertEqual(len(comps), 1)
            cuid_set = set((str(ComponentUID(comp.referent)) for comp in comps))
            self.assertEqual(cuid_set, pred_cuid_set)
        else:
            raise RuntimeError()