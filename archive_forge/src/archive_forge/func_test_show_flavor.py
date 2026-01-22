from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_show_flavor(self):
    self.requests_mock.register_uri('GET', FAKE_LBAAS_URL + 'flavors/' + FAKE_FV, json=SINGLE_FV_RESP, status_code=200)
    ret = self.api.flavor_show(FAKE_FV)
    self.assertEqual(SINGLE_FV_RESP['flavor'], ret)