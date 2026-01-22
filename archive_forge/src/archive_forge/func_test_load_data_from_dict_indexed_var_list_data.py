import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.interfaces.model_interface import DynamicModelInterface
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_load_data_from_dict_indexed_var_list_data(self):
    m = self._make_model()
    interface = DynamicModelInterface(m, m.time)
    data_list = [2, 3, 4]
    data = {pyo.ComponentUID(m.input): data_list}
    interface.load_data((data, m.time))
    for i, t in enumerate(m.time):
        self.assertEqual(m.input[t].value, data_list[i])