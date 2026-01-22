from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_stats_show_load_balancer(self):
    self.requests_mock.register_uri('GET', FAKE_LBAAS_URL + 'loadbalancers/' + FAKE_LB + '/stats', json=SINGLE_LB_STATS_RESP, status_code=200)
    ret = self.api.load_balancer_stats_show(FAKE_LB)
    self.assertEqual(SINGLE_LB_STATS_RESP, ret)