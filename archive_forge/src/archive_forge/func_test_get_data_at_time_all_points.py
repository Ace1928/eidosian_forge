import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.interfaces.model_interface import DynamicModelInterface
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_get_data_at_time_all_points(self):
    m = self._make_model()
    interface = DynamicModelInterface(m, m.time)
    data = interface.get_data_at_time(include_expr=True)
    pred_data = TimeSeriesData({m.var[:, 'A']: [1.0, 1.1, 1.2], m.var[:, 'B']: [1.0, 1.1, 1.2], m.input[:]: [1.0, 0.9, 0.8], m.var_squared[:, 'A']: [1.0 ** 2, 1.1 ** 2, 1.2 ** 2], m.var_squared[:, 'B']: [1.0 ** 2, 1.1 ** 2, 1.2 ** 2]}, m.time)
    self.assertEqual(data, pred_data)