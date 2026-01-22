from heat.common import identifier
from heat.tests import common
def test_arn_id_int(self):
    hi = identifier.HeatIdentifier('t', 's', 42, 'p')
    self.assertEqual('arn:openstack:heat::t:stacks/s/42/p', hi.arn())