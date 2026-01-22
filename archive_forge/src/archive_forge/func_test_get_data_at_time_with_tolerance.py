import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
def test_get_data_at_time_with_tolerance(self):
    m = self._make_model()
    data_dict = {m.var[:, 'A']: [1, 2, 3], m.var[:, 'B']: [2, 4, 6]}
    data = TimeSeriesData(data_dict, m.time)
    new_data = data.get_data_at_time(-0.1, tolerance=None)
    self.assertEqual(ScalarData({m.var[:, 'A']: 1, m.var[:, 'B']: 2}), new_data)
    new_data = data.get_data_at_time(-0.0001, tolerance=0.001)
    self.assertEqual(ScalarData({m.var[:, 'A']: 1, m.var[:, 'B']: 2}), new_data)
    msg = 'Time point.*is invalid'
    with self.assertRaisesRegex(RuntimeError, msg):
        new_data = data.get_data_at_time(-0.0001)
    msg = 'Time point.*is invalid'
    with self.assertRaisesRegex(RuntimeError, msg):
        new_data = data.get_data_at_time(-0.01, tolerance=0.001)