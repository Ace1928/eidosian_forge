import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_specified_index_2(self):
    """
        Components indexed only by flattened sets
        """
    m = ConcreteModel()
    m.time = Set(initialize=[1, 2, 3])
    m.space = Set(initialize=[2, 4, 6])
    m.v = Var(m.time, m.space)

    @m.Block(m.time, m.space)
    def b(b, t, x):
        b.v1 = Var()
        if x != 2:
            b.v2 = Var()
    sets = (m.time, m.space)
    sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
    self.assertEqual(len(comps_list), len(sets_list))
    self.assertEqual(len(sets_list), 1)
    for sets, comps in zip(sets_list, comps_list):
        if len(sets) == 2 and sets[0] is m.time and (sets[1] is m.space):
            self.assertEqual(len(comps), 2)
            ref_data = {self._hashRef(Reference(m.v[...])), self._hashRef(Reference(m.b[...].v1))}
            self.assertEqual(len(ref_data), len(comps))
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        else:
            raise RuntimeError()
    indices = ComponentMap([(m.space, 4)])
    sets_list, comps_list = flatten_components_along_sets(m, sets, Var, indices=indices)
    self.assertEqual(len(comps_list), len(sets_list))
    self.assertEqual(len(sets_list), 1)
    for sets, comps in zip(sets_list, comps_list):
        if len(sets) == 2 and sets[0] is m.time and (sets[1] is m.space):
            self.assertEqual(len(comps), 3)
            incomplete_slice = m.b[:, :].v2
            incomplete_slice.attribute_errors_generate_exceptions = False
            incomplete_ref = Reference(incomplete_slice)
            ref_data = {self._hashRef(Reference(m.v[:, :])), self._hashRef(Reference(m.b[:, :].v1)), self._hashRef(incomplete_ref)}
            self.assertEqual(len(ref_data), len(comps))
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        else:
            raise RuntimeError()
    indices = (3, 6)
    sets_list, comps_list = flatten_components_along_sets(m, sets, Var, indices=indices)
    self.assertEqual(len(comps_list), len(sets_list))
    self.assertEqual(len(sets_list), 1)
    for sets, comps in zip(sets_list, comps_list):
        if len(sets) == 2 and sets[0] is m.time and (sets[1] is m.space):
            self.assertEqual(len(comps), 3)
            incomplete_slice = m.b[:, :].v2
            incomplete_slice.attribute_errors_generate_exceptions = False
            incomplete_ref = Reference(incomplete_slice)
            ref_data = {self._hashRef(Reference(m.v[:, :])), self._hashRef(Reference(m.b[:, :].v1)), self._hashRef(incomplete_ref)}
            self.assertEqual(len(ref_data), len(comps))
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        else:
            raise RuntimeError()