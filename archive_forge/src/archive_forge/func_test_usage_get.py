import datetime
from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import usage
def test_usage_get(self):
    now = datetime.datetime.now()
    u = self.cs.usage.get('tenantfoo', now, now)
    self.assert_request_id(u, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('GET', '/os-simple-tenant-usage/tenantfoo?' + 'start=%s&' % now.isoformat() + 'end=%s' % now.isoformat())
    self.assertIsInstance(u, usage.Usage)