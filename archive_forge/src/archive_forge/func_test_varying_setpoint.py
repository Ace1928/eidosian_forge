import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.modeling.cost_expressions import (
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_varying_setpoint(self):
    m = self._make_model(n_time_points=5)
    A_target = [0.4, 0.6, 0.1, 0.0, 1.1]
    B_target = [0.8, 0.9, 1.3, 1.5, 1.4]
    setpoint = ({m.var[:, 'A']: A_target, m.var[:, 'B']: B_target}, m.time)
    variables = [pyo.Reference(m.var[:, 'B']), pyo.Reference(m.var[:, 'A'])]
    m.var_set, m.penalty = get_penalty_from_target(variables, m.time, setpoint)
    target = {(i, t): A_target[j] if i == 1 else B_target[t] for i in m.var_set for j, t in enumerate(m.time)}
    for i, t in m.var_set * m.time:
        pred_expr = (variables[i][t] - target[i, t]) ** 2
        self.assertTrue(compare_expressions(pred_expr, m.penalty[i, t].expr))
        self.assertEqual(pyo.value(pred_expr), pyo.value(m.penalty[i, t]))