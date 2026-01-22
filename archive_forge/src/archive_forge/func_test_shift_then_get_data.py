import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
def test_shift_then_get_data(self):
    m = self._make_model()
    data_dict = {m.var[:, 'A']: [1, 2, 3], m.var[:, 'B']: [2, 4, 6]}
    data = TimeSeriesData(data_dict, m.time)
    offset = 0.1
    data.shift_time_points(offset)
    self.assertEqual(data.get_time_points(), [t + offset for t in m.time])
    msg = 'Time point.*is invalid'
    with self.assertRaisesRegex(RuntimeError, msg):
        t0_data = data.get_data_at_time(0.0, tolerance=0.001)
    t1_data = data.get_data_at_time(0.1)
    self.assertEqual(t1_data, ScalarData({m.var[:, 'A']: 1, m.var[:, 'B']: 2}))