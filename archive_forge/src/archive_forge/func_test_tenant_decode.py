from heat.common import identifier
from heat.tests import common
def test_tenant_decode(self):
    arn = 'arn:openstack:heat::%3A%2F:stacks/s/i'
    hi = identifier.HeatIdentifier.from_arn(arn)
    self.assertEqual(':/', hi.tenant)