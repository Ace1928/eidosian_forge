from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_delete_amphora(self):
    self.requests_mock.register_uri('DELETE', FAKE_OCTAVIA_URL + 'amphorae/' + FAKE_AMP, status_code=200)
    ret = self.api.amphora_delete(FAKE_AMP)
    self.assertEqual(200, ret.status_code)