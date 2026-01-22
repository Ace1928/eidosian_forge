from heat.common import identifier
from heat.tests import common
def test_arn_parse_empty_field(self):
    arn = 'arn:openstack:heat::t:stacks//i'
    self.assertRaises(ValueError, identifier.HeatIdentifier.from_arn, arn)