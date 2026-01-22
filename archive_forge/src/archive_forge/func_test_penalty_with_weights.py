import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.modeling.cost_expressions import (
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_penalty_with_weights(self):
    m = pyo.ConcreteModel()
    m.time = pyo.Set(initialize=[1, 2, 3])
    m.v1 = pyo.Var(m.time, initialize={i: 1 * i for i in m.time})
    m.v2 = pyo.Var(m.time, initialize={i: 2 * i for i in m.time})
    setpoint_data = ScalarData({m.v1[:]: 3.0, m.v2[:]: 4.0})
    weight_data = ScalarData({m.v1[:]: 0.1, m.v2[:]: 0.5})
    m.var_set = pyo.Set(initialize=[0, 1])
    variables = [m.v1, m.v2]
    new_set, m.tracking_expr = get_penalty_from_constant_target(variables, m.time, setpoint_data, weight_data=weight_data, variable_set=m.var_set)
    self.assertIs(new_set, m.var_set)
    var_sets = {(i, t): ComponentSet(identify_variables(m.tracking_expr[i, t])) for i in m.var_set for t in m.time}
    for i in m.time:
        for j in m.var_set:
            self.assertIn(variables[j][i], var_sets[j, i])
            pred_value = 0.1 * (1 * i - 3) ** 2 if j == 0 else 0.5 * (2 * i - 4) ** 2
            self.assertAlmostEqual(pred_value, pyo.value(m.tracking_expr[j, i]))
            pred_expr = 0.1 * (m.v1[i] - 3) ** 2 if j == 0 else 0.5 * (m.v2[i] - 4) ** 2
            self.assertTrue(compare_expressions(pred_expr, m.tracking_expr[j, i].expr))