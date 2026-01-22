from keystoneauth1 import session
from osc_lib import exceptions as osc_lib_exceptions
from requests_mock.contrib import fixture
from openstackclient.api import compute_v2 as compute
from openstackclient.tests.unit import utils
def test_security_group_delete_name(self):
    self.requests_mock.register_uri('GET', FAKE_URL + '/os-security-groups/sg1', status_code=404)
    self.requests_mock.register_uri('GET', FAKE_URL + '/os-security-groups', json={'security_groups': self.LIST_SECURITY_GROUP_RESP}, status_code=200)
    self.requests_mock.register_uri('DELETE', FAKE_URL + '/os-security-groups/1', status_code=202)
    ret = self.api.security_group_delete('sg1')
    self.assertEqual(202, ret.status_code)
    self.assertEqual('', ret.text)