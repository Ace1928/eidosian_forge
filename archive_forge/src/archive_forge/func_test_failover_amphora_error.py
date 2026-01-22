from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_failover_amphora_error(self):
    self.requests_mock.register_uri('PUT', FAKE_OCTAVIA_URL + 'amphorae/' + FAKE_AMP + '/failover', text='{"faultstring": "%s"}' % self._error_message, status_code=409)
    self.assertRaisesRegex(exceptions.OctaviaClientException, self._error_message, self.api.amphora_failover, FAKE_AMP)