import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.interfaces.model_interface import DynamicModelInterface
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_get_scalar_var_data(self):
    m = self._make_model()
    interface = DynamicModelInterface(m, m.time)
    data = interface.get_scalar_variable_data()
    self.assertEqual({pyo.ComponentUID(m.scalar): 0.5}, data)