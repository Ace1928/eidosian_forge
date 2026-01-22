from keystoneauth1 import session
from osc_lib import exceptions as osc_lib_exceptions
from requests_mock.contrib import fixture
from openstackclient.api import compute_v2 as compute
from openstackclient.tests.unit import utils
def test_network_find_name(self):
    self.requests_mock.register_uri('GET', FAKE_URL + '/os-networks/label2', status_code=404)
    self.requests_mock.register_uri('GET', FAKE_URL + '/os-networks', json={'networks': self.LIST_NETWORK_RESP}, status_code=200)
    ret = self.api.network_find('label2')
    self.assertEqual(self.FAKE_NETWORK_RESP_2, ret)