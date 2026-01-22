import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.interfaces.model_interface import DynamicModelInterface
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_shift_values_by_time(self):
    m = self._make_model()
    interface = DynamicModelInterface(m, m.time)
    dt = 1.0
    interface.shift_values_by_time(dt)
    t = 0
    self.assertEqual(m.var[t, 'A'].value, 1.1)
    self.assertEqual(m.var[t, 'B'].value, 1.1)
    self.assertEqual(m.input[t].value, 0.9)
    t = 1
    self.assertEqual(m.var[t, 'A'].value, 1.2)
    self.assertEqual(m.var[t, 'B'].value, 1.2)
    self.assertEqual(m.input[t].value, 0.8)
    t = 2
    self.assertEqual(m.var[t, 'A'].value, 1.2)
    self.assertEqual(m.var[t, 'B'].value, 1.2)
    self.assertEqual(m.input[t].value, 0.8)