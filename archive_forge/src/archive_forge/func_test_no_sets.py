import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_no_sets(self):
    m = self.make_model()
    var = m.v12
    sets = (m.s3, m.s4)
    ref_data = {self._hashRef(v) for v in m.v12.values()}
    slices = [slice_ for _, slice_ in slice_component_along_sets(var, sets)]
    self.assertEqual(len(slices), len(ref_data))
    self.assertEqual(len(slices), len(m.s1) * len(m.s2))
    for slice_ in slices:
        self.assertIn(self._hashRef(slice_), ref_data)