from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_show_provider_capability(self):
    self.requests_mock.register_uri('GET', FAKE_LBAAS_URL + 'providers/' + FAKE_PROVIDER + '/flavor_capabilities', json=SINGLE_PROVIDER_CAPABILITY_RESP, status_code=200)
    ret = self.api.provider_flavor_capability_list(FAKE_PROVIDER)
    self.assertEqual(SINGLE_PROVIDER_CAPABILITY_RESP, ret)