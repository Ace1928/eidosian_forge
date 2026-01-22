from keystoneauth1 import session
from osc_lib import exceptions as osc_lib_exceptions
from requests_mock.contrib import fixture
from openstackclient.api import compute_v2 as compute
from openstackclient.tests.unit import utils
def test_floating_ip_pool_list(self):
    self.requests_mock.register_uri('GET', FAKE_URL + '/os-floating-ip-pools', json={'floating_ip_pools': self.LIST_FLOATING_IP_POOL_RESP}, status_code=200)
    ret = self.api.floating_ip_pool_list()
    self.assertEqual(self.LIST_FLOATING_IP_POOL_RESP, ret)