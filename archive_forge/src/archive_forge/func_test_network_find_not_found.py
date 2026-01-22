from keystoneauth1 import session
from osc_lib import exceptions as osc_lib_exceptions
from requests_mock.contrib import fixture
from openstackclient.api import compute_v2 as compute
from openstackclient.tests.unit import utils
def test_network_find_not_found(self):
    self.requests_mock.register_uri('GET', FAKE_URL + '/os-networks/label3', status_code=404)
    self.requests_mock.register_uri('GET', FAKE_URL + '/os-networks', json={'networks': self.LIST_NETWORK_RESP}, status_code=200)
    self.assertRaises(osc_lib_exceptions.NotFound, self.api.network_find, 'label3')