from keystoneauth1 import session
from osc_lib import exceptions as osc_lib_exceptions
from requests_mock.contrib import fixture
from openstackclient.api import compute_v2 as compute
from openstackclient.tests.unit import utils
def test_network_delete_id(self):
    self.requests_mock.register_uri('GET', FAKE_URL + '/os-networks/1', json={'network': self.FAKE_NETWORK_RESP}, status_code=200)
    self.requests_mock.register_uri('DELETE', FAKE_URL + '/os-networks/1', status_code=202)
    ret = self.api.network_delete('1')
    self.assertEqual(202, ret.status_code)
    self.assertEqual('', ret.text)