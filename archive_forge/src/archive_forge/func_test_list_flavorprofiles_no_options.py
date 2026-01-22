from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_list_flavorprofiles_no_options(self):
    self.requests_mock.register_uri('GET', FAKE_LBAAS_URL + 'flavorprofiles', json=LIST_FVPF_RESP, status_code=200)
    ret = self.api.flavorprofile_list()
    self.assertEqual(LIST_FVPF_RESP, ret)