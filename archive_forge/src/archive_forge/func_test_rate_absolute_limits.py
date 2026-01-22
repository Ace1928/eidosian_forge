from novaclient import api_versions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import limits as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import limits
def test_rate_absolute_limits(self):
    obj = self.cs.limits.get()
    self.assert_request_id(obj, fakes.FAKE_REQUEST_ID_LIST)
    expected = (limits.RateLimit('POST', '*', '.*', 10, 2, 'MINUTE', '2011-12-15T22:42:45Z'), limits.RateLimit('PUT', '*', '.*', 10, 2, 'MINUTE', '2011-12-15T22:42:45Z'), limits.RateLimit('DELETE', '*', '.*', 100, 100, 'MINUTE', '2011-12-15T22:42:45Z'), limits.RateLimit('POST', '*/servers', '^/servers', 25, 24, 'DAY', '2011-12-15T22:42:45Z'))
    rate_limits = list(obj.rate)
    self.assertEqual(len(rate_limits), len(expected))
    for limit in rate_limits:
        self.assertIn(limit, expected)
    expected = [limits.AbsoluteLimit('maxTotalRAMSize', 51200), limits.AbsoluteLimit('maxServerMeta', 5)]
    if self.supports_image_meta:
        expected.append(limits.AbsoluteLimit('maxImageMeta', 5))
    if self.supports_personality:
        expected.extend([limits.AbsoluteLimit('maxPersonality', 5), limits.AbsoluteLimit('maxPersonalitySize', 10240)])
    abs_limits = list(obj.absolute)
    self.assertEqual(len(abs_limits), len(expected))
    for limit in abs_limits:
        self.assertIn(limit, expected)