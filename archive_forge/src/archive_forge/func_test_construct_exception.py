import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.mpc.data.scalar_data import ScalarData
def test_construct_exception(self):
    m = self._make_model()
    msg = 'Value.*not a scalar'
    with self.assertRaisesRegex(TypeError, msg):
        data = ScalarData({m.var[:, 'A']: [1, 2]})