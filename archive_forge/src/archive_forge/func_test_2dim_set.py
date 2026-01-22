import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_2dim_set(self):
    m = ConcreteModel()
    m.time = ContinuousSet(bounds=(0, 1))
    m.v = Var(m.time, [('a', 1), ('b', 2)])
    scalar, dae = flatten_dae_components(m, m.time, Var)
    self.assertEqual(len(scalar), 0)
    ref_data = {self._hashRef(Reference(m.v[:, 'a', 1])), self._hashRef(Reference(m.v[:, 'b', 2]))}
    self.assertEqual(len(dae), len(ref_data))
    for ref in dae:
        self.assertIn(self._hashRef(ref), ref_data)