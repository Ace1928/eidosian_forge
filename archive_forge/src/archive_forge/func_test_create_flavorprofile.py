from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_create_flavorprofile(self):
    self.requests_mock.register_uri('POST', FAKE_LBAAS_URL + 'flavorprofiles', json=SINGLE_FVPF_RESP, status_code=200)
    ret = self.api.flavorprofile_create(json=SINGLE_FVPF_RESP)
    self.assertEqual(SINGLE_FVPF_RESP, ret)