import datetime
from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import usage
def test_usage_list_with_paging(self):
    now = datetime.datetime.now()
    usages = self.cs.usage.list(now, now, marker='some-uuid', limit=3)
    self.assert_request_id(usages, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('GET', '/os-simple-tenant-usage?' + 'start=%s&' % now.isoformat() + 'end=%s&' % now.isoformat() + 'limit=3&marker=some-uuid&detailed=0')
    for u in usages:
        self.assertIsInstance(u, usage.Usage)