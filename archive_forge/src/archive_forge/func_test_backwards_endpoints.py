import pyomo.common.unittest as unittest
import pytest
import pyomo.environ as pyo
import pyomo.dae as dae
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.interval_data import assert_disjoint_intervals, IntervalData
def test_backwards_endpoints(self):
    intervals = [(0, 1), (3, 2)]
    msg = 'Lower endpoint of interval is higher'
    with self.assertRaisesRegex(RuntimeError, msg):
        assert_disjoint_intervals(intervals)