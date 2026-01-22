from heat.common import identifier
from heat.tests import common
def test_arn(self):
    hi = identifier.HeatIdentifier('t', 's', 'i', 'p')
    self.assertEqual('arn:openstack:heat::t:stacks/s/i/p', hi.arn())