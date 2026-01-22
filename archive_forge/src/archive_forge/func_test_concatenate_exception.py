import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
def test_concatenate_exception(self):
    m = self._make_model()
    data_dict = {m.var[:, 'A']: [1, 2, 3], m.var[:, 'B']: [2, 4, 6]}
    time1 = [t for t in m.time]
    data1 = TimeSeriesData(data_dict, time1)
    msg = 'Initial time point.*is not greater'
    with self.assertRaisesRegex(ValueError, msg):
        data1.concatenate(data1)