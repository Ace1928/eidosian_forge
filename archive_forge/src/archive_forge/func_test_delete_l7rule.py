from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_delete_l7rule(self):
    self.requests_mock.register_uri('DELETE', FAKE_LBAAS_URL + 'l7policies/' + FAKE_L7PO + '/rules/' + FAKE_L7RU, status_code=200)
    ret = self.api.l7rule_delete(l7rule_id=FAKE_L7RU, l7policy_id=FAKE_L7PO)
    self.assertEqual(200, ret.status_code)