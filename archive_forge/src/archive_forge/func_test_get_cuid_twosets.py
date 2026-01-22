import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.mpc.data.get_cuid import get_indexed_cuid
def test_get_cuid_twosets(self):
    m = self._make_model()
    pred_cuid = pyo.ComponentUID(m.b[:, :].bvar2['A'])
    self.assertEqual(get_indexed_cuid(m.b[:, :].bvar2['A']), pred_cuid)
    self.assertEqual(get_indexed_cuid(pyo.Reference(m.b[:, :].bvar2['A'])), pred_cuid)
    self.assertEqual(get_indexed_cuid('b[*,*].bvar2[A]'), pred_cuid)
    self.assertEqual(get_indexed_cuid(m.b[0, 1].bvar2['A'], sets=(m.time, m.space)), pred_cuid)