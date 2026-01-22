from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_set_l7rule_error(self):
    self.requests_mock.register_uri('PUT', FAKE_LBAAS_URL + 'l7policies/' + FAKE_L7PO + '/rules/' + FAKE_L7RU, text='{"faultstring": "%s"}' % self._error_message, status_code=400)
    self.assertRaisesRegex(exceptions.OctaviaClientException, self._error_message, self.api.l7rule_set, l7rule_id=FAKE_L7RU, l7policy_id=FAKE_L7PO, json=SINGLE_L7RU_UPDATE)