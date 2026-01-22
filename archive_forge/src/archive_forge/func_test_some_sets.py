import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_some_sets(self):
    m = self.make_model()
    var = m.v124
    sets = (m.s1, m.s3)
    ref_data = {self._hashRef(Reference(m.v124[:, i, j])) for i, j in m.s2 * m.s4}
    slices = [s for _, s in slice_component_along_sets(var, sets)]
    self.assertEqual(len(slices), len(ref_data))
    self.assertEqual(len(slices), len(m.s2) * len(m.s4))
    for slice_ in slices:
        self.assertIn(self._hashRef(Reference(slice_)), ref_data)