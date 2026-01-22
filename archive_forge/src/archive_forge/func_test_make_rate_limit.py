from openstack.block_storage.v2 import limits
from openstack.tests.unit import base
def test_make_rate_limit(self):
    limit_resource = limits.RateLimit(**RATE_LIMIT)
    self.assertEqual(RATE_LIMIT['verb'], limit_resource.verb)
    self.assertEqual(RATE_LIMIT['value'], limit_resource.value)
    self.assertEqual(RATE_LIMIT['remaining'], limit_resource.remaining)
    self.assertEqual(RATE_LIMIT['unit'], limit_resource.unit)
    self.assertEqual(RATE_LIMIT['next-available'], limit_resource.next_available)