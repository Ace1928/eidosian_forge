from keystoneauth1 import session
from osc_lib import exceptions as osc_lib_exceptions
from requests_mock.contrib import fixture
from openstackclient.api import compute_v2 as compute
from openstackclient.tests.unit import utils
def test_floating_ip_find_ip(self):
    self.requests_mock.register_uri('GET', FAKE_URL + '/os-floating-ips/' + self.FAKE_FLOATING_IP_RESP['ip'], status_code=404)
    self.requests_mock.register_uri('GET', FAKE_URL + '/os-floating-ips', json={'floating_ips': self.LIST_FLOATING_IP_RESP}, status_code=200)
    ret = self.api.floating_ip_find(self.FAKE_FLOATING_IP_RESP['ip'])
    self.assertEqual(self.FAKE_FLOATING_IP_RESP, ret)