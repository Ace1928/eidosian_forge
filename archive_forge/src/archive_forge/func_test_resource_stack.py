from heat.common import identifier
from heat.tests import common
def test_resource_stack(self):
    si = identifier.HeatIdentifier('t', 's', 'i')
    ri = identifier.ResourceIdentifier(resource_name='r', **si)
    self.assertEqual(si, ri.stack())