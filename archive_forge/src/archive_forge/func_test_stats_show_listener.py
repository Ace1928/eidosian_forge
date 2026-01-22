from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_stats_show_listener(self):
    self.requests_mock.register_uri('GET', FAKE_LBAAS_URL + 'listeners/' + FAKE_LI + '/stats', json=SINGLE_LB_STATS_RESP, status_code=200)
    ret = self.api.listener_stats_show(FAKE_LI)
    self.assertEqual(SINGLE_LB_STATS_RESP, ret)