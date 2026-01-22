import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.modeling.cost_expressions import (
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_varying_setpoint_exceptions(self):
    m = self._make_model(n_time_points=5)
    variables = [pyo.Reference(m.var[:, 'A']), pyo.Reference(m.var[:, 'B'])]
    A_setpoint = [1.0 - 0.1 * i for i in range(len(m.time))]
    B_setpoint = [5.0 + 0.1 * i for i in range(len(m.time))]
    setpoint_data = TimeSeriesData({m.var[:, 'A']: A_setpoint, m.var[:, 'B']: B_setpoint}, [i + 10 for i in m.time])
    weight_data = {pyo.ComponentUID(m.var[:, 'A']): 10.0, pyo.ComponentUID(m.var[:, 'B']): 0.1}
    msg = 'Mismatch in time points'
    with self.assertRaisesRegex(RuntimeError, msg):
        var_set, tr_cost = get_penalty_from_time_varying_target(variables, m.time, setpoint_data, weight_data=weight_data)
    setpoint_data = TimeSeriesData({m.var[:, 'A']: A_setpoint}, m.time)
    msg = 'Setpoint data does not contain'
    with self.assertRaisesRegex(KeyError, msg):
        var_set, tr_cost = get_penalty_from_time_varying_target(variables, m.time, setpoint_data, weight_data=weight_data)
    setpoint_data = TimeSeriesData({m.var[:, 'A']: A_setpoint, m.var[:, 'B']: B_setpoint}, m.time)
    weight_data = {pyo.ComponentUID(m.var[:, 'A']): 10.0}
    msg = 'Tracking weight does not contain'
    with self.assertRaisesRegex(KeyError, msg):
        tr_cost = get_penalty_from_time_varying_target(variables, m.time, setpoint_data, weight_data=weight_data)