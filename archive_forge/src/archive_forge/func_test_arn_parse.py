from heat.common import identifier
from heat.tests import common
def test_arn_parse(self):
    arn = 'arn:openstack:heat::t:stacks/s/i/p'
    hi = identifier.HeatIdentifier.from_arn(arn)
    self.assertEqual('t', hi.tenant)
    self.assertEqual('s', hi.stack_name)
    self.assertEqual('i', hi.stack_id)
    self.assertEqual('/p', hi.path)