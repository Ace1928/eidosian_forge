from heat.common import identifier
from heat.tests import common
def test_name_decode(self):
    arn = 'arn:openstack:heat::t:stacks/%3A%25/i'
    hi = identifier.HeatIdentifier.from_arn(arn)
    self.assertEqual(':%', hi.stack_name)