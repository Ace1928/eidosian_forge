import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.mpc.data.scalar_data import ScalarData
def test_to_serializable(self):
    m = self._make_model()
    data = ScalarData({m.var[:, 'A']: 0.5, m.var[:, 'B']: 2.0})
    pred_json_dict = {'var[*,A]': 0.5, 'var[*,B]': 2.0}
    self.assertEqual(data.to_serializable(), pred_json_dict)