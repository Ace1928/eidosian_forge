from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_show_availabilityzone(self):
    self.requests_mock.register_uri('GET', FAKE_LBAAS_URL + 'availabilityzones/' + FAKE_AZ, json=SINGLE_AZ_RESP, status_code=200)
    ret = self.api.availabilityzone_show(FAKE_AZ)
    self.assertEqual(SINGLE_AZ_RESP['availability_zone'], ret)