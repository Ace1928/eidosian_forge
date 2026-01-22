import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.interfaces.model_interface import DynamicModelInterface
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_load_data_from_ScalarData_tosubset(self):
    m = self._make_model()
    interface = DynamicModelInterface(m, m.time)
    old_A = {t: m.var[t, 'A'].value for t in m.time}
    old_input = {t: m.input[t].value for t in m.time}
    data = ScalarData({m.var[:, 'A']: 5.5, m.input[:]: 6.6})
    time_points = [1, 2]
    time_set = set(time_points)
    interface.load_data(data, time_points=[1, 2])
    B_data = [m.var[t, 'B'].value for t in m.time]
    self.assertEqual(B_data, [1.0, 1.1, 1.2])
    for t in m.time:
        if t in time_set:
            self.assertEqual(m.var[t, 'A'].value, 5.5)
            self.assertEqual(m.input[t].value, 6.6)
        else:
            self.assertEqual(m.var[t, 'A'].value, old_A[t])
            self.assertEqual(m.input[t].value, old_input[t])