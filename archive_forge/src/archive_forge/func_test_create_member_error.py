from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_create_member_error(self):
    self.requests_mock.register_uri('POST', FAKE_LBAAS_URL + 'pools/' + FAKE_PO + '/members', text='{"faultstring": "%s"}' % self._error_message, status_code=400)
    self.assertRaisesRegex(exceptions.OctaviaClientException, self._error_message, self.api.member_create, json=SINGLE_ME_RESP, pool_id=FAKE_PO)