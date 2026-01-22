from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_show_health_monitor(self):
    self.requests_mock.register_uri('GET', FAKE_LBAAS_URL + 'healthmonitors/' + FAKE_HM, json=SINGLE_HM_RESP, status_code=200)
    ret = self.api.health_monitor_show(FAKE_HM)
    self.assertEqual(SINGLE_HM_RESP['healthmonitor'], ret)