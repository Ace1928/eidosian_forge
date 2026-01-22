from keystoneauth1 import session
from oslo_utils import uuidutils
from requests_mock.contrib import fixture
from osc_lib.tests import utils
from octaviaclient.api import exceptions
from octaviaclient.api.v2 import octavia
def test_failover_load_balancer(self):
    self.requests_mock.register_uri('PUT', FAKE_LBAAS_URL + 'loadbalancers/' + FAKE_LB + '/failover', status_code=202)
    ret = self.api.load_balancer_failover(FAKE_LB)
    self.assertEqual(202, ret.status_code)