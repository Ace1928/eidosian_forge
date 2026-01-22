import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.modeling.cost_expressions import (
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
def test_bad_data_tuple_entry_0(self):
    m = self._make_model(n_time_points=3)
    A_target = [0.4, 0.6, 0.1]
    B_target = [0.8, 0.9, 1.3]
    setpoint = ([(m.var[:, 'A'], A_target), (m.var[:, 'B'], B_target)], m.time)
    variables = [pyo.Reference(m.var[:, 'B']), pyo.Reference(m.var[:, 'A'])]
    msg = 'must be instance of MutableMapping'
    with self.assertRaisesRegex(TypeError, msg):
        m.var_set, m.penalty = get_penalty_from_target(variables, m.time, setpoint)