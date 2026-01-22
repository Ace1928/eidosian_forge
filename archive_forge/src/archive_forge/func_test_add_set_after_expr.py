import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.modeling.cost_expressions import (
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_add_set_after_expr(self):
    m = pyo.ConcreteModel()
    m.time = pyo.Set(initialize=[1, 2, 3])
    m.v1 = pyo.Var(m.time, initialize={i: 1 * i for i in m.time})
    m.v2 = pyo.Var(m.time, initialize={i: 2 * i for i in m.time})
    setpoint_data = ScalarData({m.v1[:]: 3.0, m.v2[:]: 4.0})
    weight_data = ScalarData({m.v1[:]: 0.1, m.v2[:]: 0.5})
    m.var_set = pyo.Set(initialize=[0, 1])
    variables = [m.v1, m.v2]
    new_set, tr_expr = get_penalty_from_constant_target(variables, m.time, setpoint_data, weight_data=weight_data, variable_set=m.var_set)
    m.tracking_expr = tr_expr
    msg = 'Attempting to re-assign'
    with self.assertRaisesRegex(RuntimeError, msg):
        m.variable_set = new_set