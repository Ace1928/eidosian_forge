import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_model4_xyz(self):
    m = self._model4_three_1d_sets()
    sets = (m.X, m.Y, m.Z)
    sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
    self.assertEqual(len(comps_list), len(sets_list))
    self.assertEqual(len(sets_list), 3)
    for sets, comps in zip(sets_list, comps_list):
        if len(sets) == 1 and sets[0] is UnindexedComponent_set:
            ref_data = {self._hashRef(Reference(m.u))}
            self.assertEqual(len(ref_data), len(comps))
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        elif len(sets) == 2 and sets[0] is m.X and (sets[1] is m.Y):
            ref_data = {self._hashRef(Reference(m.base[:, :])), self._hashRef(Reference(m.b2[:, :].base))}
            self.assertEqual(len(ref_data), len(comps))
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        elif len(sets) == 3 and sets[0] is m.X and (sets[1] is m.Y) and (sets[2] is m.Z):
            ref_data = {self._hashRef(Reference(m.v[:, :, :, 'a'])), self._hashRef(Reference(m.v[:, :, :, 'b'])), self._hashRef(Reference(m.b4[:, :, :, 'a'].v)), self._hashRef(Reference(m.b4[:, :, :, 'b'].v)), self._hashRef(Reference(m.b2[:, :].v[:, 'a'])), self._hashRef(Reference(m.b2[:, :].v[:, 'b']))}
            self.assertEqual(len(ref_data), len(comps))
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        else:
            raise RuntimeError()