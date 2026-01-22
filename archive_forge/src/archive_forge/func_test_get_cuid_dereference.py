import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.mpc.data.get_cuid import get_indexed_cuid
def test_get_cuid_dereference(self):
    m = self._make_model()
    m.ref = pyo.Reference(m.var[:, 'A'])
    m.ref2 = pyo.Reference(m.ref)
    pred_cuid = pyo.ComponentUID(m.var[:, 'A'])
    self.assertNotEqual(get_indexed_cuid(m.ref), pred_cuid)
    self.assertEqual(get_indexed_cuid(m.ref), pyo.ComponentUID(m.ref[:]))
    self.assertEqual(get_indexed_cuid(m.ref, dereference=True), pred_cuid)
    self.assertNotEqual(get_indexed_cuid(m.ref2, dereference=True), pred_cuid)
    self.assertEqual(get_indexed_cuid(m.ref2, dereference=True), pyo.ComponentUID(m.ref[:]))
    self.assertEqual(get_indexed_cuid(m.ref2, dereference=2), pred_cuid)