import pyomo.common.unittest as unittest
import pytest
import pyomo.environ as pyo
import pyomo.dae as dae
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import assert_disjoint_intervals, IntervalData
def test_get_data_at_time_scalar(self):
    m = self._make_model()
    intervals = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
    data = {m.var[:, 'A']: [1.0, 2.0, 3.0], m.var[:, 'B']: [4.0, 5.0, 6.0]}
    interval_data = IntervalData(data, intervals)
    data = interval_data.get_data_at_time(0.1)
    pred_data = ScalarData({m.var[:, 'A']: 1.0, m.var[:, 'B']: 4.0})
    self.assertEqual(data, pred_data)
    data = interval_data.get_data_at_time(1.1)
    pred_data = ScalarData({m.var[:, 'A']: 3.0, m.var[:, 'B']: 6.0})
    self.assertEqual(data, pred_data)
    msg = 'Time point.*not found'
    with self.assertRaisesRegex(RuntimeError, msg):
        data = interval_data.get_data_at_time(1.1, tolerance=0.001)
    data = interval_data.get_data_at_time(0.5)
    pred_data = ScalarData({m.var[:, 'A']: 2.0, m.var[:, 'B']: 5.0})
    self.assertEqual(data, pred_data)
    data = interval_data.get_data_at_time(0.5, prefer_left=False)
    pred_data = ScalarData({m.var[:, 'A']: 3.0, m.var[:, 'B']: 6.0})
    self.assertEqual(data, pred_data)