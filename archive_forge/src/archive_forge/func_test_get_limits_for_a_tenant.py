from novaclient import api_versions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import limits as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import limits
def test_get_limits_for_a_tenant(self):
    obj = self.cs.limits.get(tenant_id=1234)
    self.assert_request_id(obj, fakes.FAKE_REQUEST_ID_LIST)
    self.assert_called('GET', '/limits?tenant_id=1234')
    self.assertIsInstance(obj, limits.Limits)