from heat.common import identifier
from heat.tests import common
def test_arn_url_parse_arn_invalid(self):
    url = self.url_prefix + 'urn:openstack:heat::t:stacks/s/i/p'
    self.assertRaises(ValueError, identifier.HeatIdentifier.from_arn_url, url)