import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.preprocessing.plugins.var_aggregator import (
from pyomo.environ import (
def test_var_aggregate(self):
    """Test for transitivity in a variable equality set."""
    m = self.build_model()
    TransformationFactory('contrib.aggregate_vars').apply_to(m)
    z_to_vars = m._var_aggregator_info.z_to_vars
    var_to_z = m._var_aggregator_info.var_to_z
    z = m._var_aggregator_info.z
    self.assertEqual(z_to_vars[z[1]], ComponentSet([m.v3, m.v4]))
    self.assertEqual(z_to_vars[z[2]], ComponentSet([m.x[1], m.x[2], m.x[3], m.x[4]]))
    self.assertEqual(z_to_vars[z[3]], ComponentSet([m.y[1], m.y[2]]))
    self.assertIs(var_to_z[m.v3], z[1])
    self.assertIs(var_to_z[m.v4], z[1])
    self.assertIs(var_to_z[m.x[1]], z[2])
    self.assertIs(var_to_z[m.x[2]], z[2])
    self.assertIs(var_to_z[m.x[3]], z[2])
    self.assertIs(var_to_z[m.x[4]], z[2])
    self.assertIs(var_to_z[m.y[1]], z[3])
    self.assertIs(var_to_z[m.y[2]], z[3])
    self.assertEqual(z[1].value, 2)
    self.assertEqual(z[1].lb, 2)
    self.assertEqual(z[1].ub, 4)
    self.assertEqual(z[3].value, 3.5)