from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_create_l7rule(self):
    self.requests_mock.register_uri('POST', FAKE_LBAAS_URL + 'l7policies/' + FAKE_L7PO + '/rules', json=SINGLE_L7RU_RESP, status_code=200)
    ret = self.api.l7rule_create(FAKE_L7PO, json=SINGLE_L7RU_RESP)
    self.assertEqual(SINGLE_L7RU_RESP, ret)