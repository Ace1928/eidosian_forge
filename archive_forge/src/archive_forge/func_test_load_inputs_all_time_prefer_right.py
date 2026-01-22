import pyomo.common.unittest as unittest
import pytest
import pyomo.environ as pyo
import pyomo.dae as dae
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import assert_disjoint_intervals, IntervalData
def test_load_inputs_all_time_prefer_right(self):
    m = self.make_model()
    interface = mpc.DynamicModelInterface(m, m.time)
    inputs = mpc.IntervalData({'v': [1.0, 2.0]}, [(0, 3), (3, 6)])
    interface.load_data(inputs, prefer_left=False)
    for t in m.time:
        if t < 3:
            self.assertEqual(m.v[t].value, 1.0)
        elif t == 6:
            self.assertEqual(m.v[t].value, 0.0)
        else:
            self.assertEqual(m.v[t].value, 2.0)