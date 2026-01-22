import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.modeling.cost_expressions import (
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_piecewise_constant_setpoint(self):
    m = self._make_model(n_time_points=5)
    A_target = [0.3, 0.9, 0.7]
    B_target = [1.1, 0.1, 0.5]
    setpoint = ({m.var[:, 'A']: A_target, m.var[:, 'B']: B_target}, [(0.0, 0.0), (0.0, 2.0), (2.0, 4.0)])
    variables = [pyo.Reference(m.var[:, 'B']), pyo.Reference(m.var[:, 'A'])]
    m.var_set, m.penalty = get_penalty_from_target(variables, m.time, setpoint)
    target = {(i, j): A_target[j] if i == 1 else B_target[j] for i in m.var_set for j in range(len(A_target))}
    for i, t in m.var_set * m.time:
        if t == 0:
            idx = 0
        elif t <= 2.0:
            idx = 1
        elif t <= 4.0:
            idx = 2
        pred_expr = (variables[i][t] - target[i, idx]) ** 2
        self.assertTrue(compare_expressions(pred_expr, m.penalty[i, t].expr))
        self.assertEqual(pyo.value(pred_expr), pyo.value(m.penalty[i, t]))