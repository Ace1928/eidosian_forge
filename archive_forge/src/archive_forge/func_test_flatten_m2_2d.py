import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_flatten_m2_2d(self):
    """
        This test has some issues due to incompatibility between
        slicing and `normalize_index.flatten==False`.
        """
    m = self._model2_nd_sets()
    sets = ComponentSet((m.d2,))
    normalize_index.flatten = False
    sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
    ref1 = Reference(m.v_2n[:, ('c', 3)])
    ref_set = ref1.index_set()._ref
    self.assertNotIn(('a', 1), ref_set)
    self.assertEqual(len(sets_list), len(comps_list))
    self.assertEqual(len(sets_list), 2)
    normalize_index.flatten = True