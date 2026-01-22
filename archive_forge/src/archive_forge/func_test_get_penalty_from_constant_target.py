import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.interfaces.model_interface import DynamicModelInterface
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_get_penalty_from_constant_target(self):
    m = self._make_model()
    interface = DynamicModelInterface(m, m.time)
    setpoint_data = ScalarData({m.var[:, 'A']: 1.0, m.var[:, 'B']: 2.0})
    weight_data = ScalarData({m.var[:, 'A']: 10.0, m.var[:, 'B']: 0.1})
    vset, tr_cost = interface.get_penalty_from_target(setpoint_data, weight_data=weight_data)
    m.var_set = vset
    m.tracking_cost = tr_cost
    for t in m.time:
        for i in m.var_set:
            pred_expr = 10.0 * (m.var[t, 'A'] - 1.0) ** 2 if i == 0 else 0.1 * (m.var[t, 'B'] - 2.0) ** 2
            self.assertEqual(pyo.value(pred_expr), pyo.value(m.tracking_cost[i, t]))
            self.assertTrue(compare_expressions(pred_expr, m.tracking_cost[i, t].expr))