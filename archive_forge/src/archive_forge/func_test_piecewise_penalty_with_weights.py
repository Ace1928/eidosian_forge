import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.modeling.cost_expressions import (
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_piecewise_penalty_with_weights(self):
    m = self._make_model(n_time_points=5)
    variables = [pyo.Reference(m.var[:, 'A']), pyo.Reference(m.var[:, 'B'])]
    setpoint_data = IntervalData({m.var[:, 'A']: [2.0, 2.5], m.var[:, 'B']: [3.0, 3.5]}, [(0, 2), (2, 4)])
    weight_data = {pyo.ComponentUID(m.var[:, 'A']): 10.0, pyo.ComponentUID(m.var[:, 'B']): 0.1}
    m.var_set, m.tracking_cost = get_penalty_from_piecewise_constant_target(variables, m.time, setpoint_data, weight_data=weight_data)
    for i in m.time:
        for j in m.var_set:
            if i <= 2:
                pred_expr = 10.0 * (m.var[i, 'A'] - 2.0) ** 2 if j == 0 else 0.1 * (m.var[i, 'B'] - 3.0) ** 2
            else:
                pred_expr = 10.0 * (m.var[i, 'A'] - 2.5) ** 2 if j == 0 else 0.1 * (m.var[i, 'B'] - 3.5) ** 2
            pred_value = pyo.value(pred_expr)
            self.assertEqual(pred_value, pyo.value(m.tracking_cost[j, i]))
            self.assertTrue(compare_expressions(pred_expr, m.tracking_cost[j, i].expr))