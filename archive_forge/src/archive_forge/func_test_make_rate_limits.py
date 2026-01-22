from openstack.block_storage.v2 import limits
from openstack.tests.unit import base
def test_make_rate_limits(self):
    limit_resource = limits.RateLimits(**RATE_LIMITS)
    self.assertEqual(RATE_LIMITS['regex'], limit_resource.regex)
    self.assertEqual(RATE_LIMITS['uri'], limit_resource.uri)
    self._test_rate_limit(RATE_LIMITS['limit'], limit_resource.limits)